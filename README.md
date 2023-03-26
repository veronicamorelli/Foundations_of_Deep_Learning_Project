# Foundations of Deep Learning Project
Rice grain image classification with CNNs

To develop the project we use the dataset available on Kaggle at the following link: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset. The dataset is called *Rice Image Dataset*. It contains 75K images including 15K images for each rice variety that are: Arborio, Basmati, Ipsala, Jasmine and Karacadag.

<img src="images/rice_grain.PNG" width=600 height=150>

We decide to use the dataset to solve a **classification problem**. We want to find a performing deep learning model to correctly classify 5 types of rice.

We mainly use two approaches:

*  **Convolutional Neaural Network from scratch**
*  **Transfer learning and Fine Tuning and feature extraction**

In the first case we design a new architecture of a Convolutional Neural Network from scratch. <br>
In the second case we use completely a Pre-trained Neural Network adding a Feed-Forward Neural Network to improve the performances.  <br>
Neural networks with the best performances are presented in this notebook. <br>
To finally choose the best model, we created a new dataset, called *test* on which we fitted the two models. The model with better classificatory goodness-of-fit is chosen as the best model. 

## Main Steps

- Understanding of the Dataset
- Loading and Sampling
- EDA: Exploratory Data Analysis
- Preprocessing + Splitting data into Training and Validation Set
-  Classification: Neural Network from scratch and Transfer Learning
- Selection and Comparison of the Best Models
- Fitting on Test data and Best Model Choice

## Main Results CNNs from Scratch

| Model Name  | Network Architecure + Properties | Accuracy Train | Accuracy Validation | Notes        	                          |
| ----------- | -------------------------------- | -------------- | ------------------- | --------------------------------------- |
| Modelv1 | 2CNN (32, 64) | 0.95 | 0.88 | From the 30th epoch onward the model remains stable. Despite having very good performance and there seems to be no presence of overfitting it is decided to increase the stability of the model in the next steps. |
| Modelv2 | 2CNN (32, 64) batch_norm. | 0.97 | 0.88 | Very unstable model. <br> The complexity of the convolutional layer is increased. |
| Modelv3 | 3 CNN (32, 64, 128) + batch_norm. | 0.98 | 0.86 | Very unstable model. Accuracy decreased. <br> The regularization term is added. |
| Modelv4l1 | 3 CNN (32, 64, 128) + batch_norm + lasso_regular. | 0.97 | 0.65 | Very unstable model. As in the previous cases the Accuracy (on training) is very high while on validation  |
| Modelv4l2 | 3 CNN (32, 64, 128) + batch_norm + ridge_regular. | 0.98 | 0.88 | Accuracy on validation is higher than that of the previous model (lasso regularization). <br> The model is more sable in the last 5 epochs (35th onwards). <br> Complexity in the fully connected part is increased. |
| Modelv4l2_5 | 3 CNN (32, 64, 128) + batch_norm + ridge_regular. + early_stop. | 0.97 | 0.92 | Accuracy is increased. |
| Modelv4_5 | 3 CNN (32, 64, 128) + batch_norm + 1 FC (64) | 0.98 | 0.81 | Very unstable model. Accuracy on validation has decreased significantly compared to that obtained in *Modelv412*. <br> However, it is tried to increase the complexity of the fully connected layer again. |
| Modelv5 | 3 CNN (32, 64, 128) + batch_norm + 2 FC (128, 64) | 0.98 | 0.82 | Accuracy has improved by 1%. Despite this improvement the performance of *Modelv412* (with only output dense layer) seems to be better. <br> We change the weight initialization algorithm in the convolutional layers. <br> We insert the He initializer because it might work better in convolutional layers where the activation function is a ReLU. |
| Modelv6 | 3 CNN (32, 64, 128) + batch_norm + ridge_regular. + He weight inizialization | 0.98 | 0.93 | The algorithm converges very early. The final performance at the 40th epoch is very high, compared with previous networks. Nevertheless, it turns out to be somewhat unstable in the later epochs. <br> Early Stopping mechanism is implemented. |
| Model v6_5 | 3 CNN (32, 64, 128) + batch_norm + ridge_regular. + He weight inizialization + early_stop | 0.98 | 0.95 | The reducing learning rate mechanism is added to try to make the model more stable and increase its performance. |
| Model v7 | 3 CNN (32, 64, 128) + batch_norm + ridge_regular. + He weight inizialization + Reduce learning rate | 0.996 | 0.98 | The model achieves very good performance on both training and validation. In addition, the model is very stable in the 20th epoch onwards. |

## Transfer Learning

### Steps: 

- Preprocess Data: The images are converted from RGB to BGR, then  each color channel is zero-centered with respect to the ImageNet  dataset, without scaling.
- Loading of the pre-trained network.
- Freeze the convolutional base before compiling and train the  model.
- Define a FNN architecture.
- Callbacks.
- Compile and train.

**Cut T1 and T2 on VGG16**

<img src="images/Vgg16_cut.png" width=450 height=250>

## Main Results

| Model Name  | Network Architecure + Properties | Accuracy Train | Accuracy Validation | Notes        	                          |
| ----------- | -------------------------------- | -------------- | ------------------- | --------------------------------------- |
| Resnet50 | Pre-trained Network+ FFN + Dropout + CallBacks (EarlyStopping, ReduceLrOnPlateau) | 0.97 | 0.98 | The network performs very well and it converges at the 14th epoch without ever overfitting. |
| MobileNet | Pre-trained Network + FFN+Droput + CallBacks (EarlyStopping, ReduceLrOnPlateau) | 0.88 | 0.85 | The performance is slightly worse than the previous network. In addition, the network tends to overfit. |
| Vgg16 | Pre-trained Network + FFN+Droput + CallBacks (EarlyStopping, ReduceLrOnPlateau) | 0.83 | 0.85 | Although the performance is the worst among trained networks, the trend of loss and accuracy curves is excellent. |
| Vgg16 cut1 | Vgg 16 cut1 Pre-trained Network + Avg global pooling + FFN + Dropout + CallBacks (EarlyStopping, ReduceLrOnPlateau) | 0.87 | 0.95 | Accuracy values are improved with a cut to block3_pool. |
| Vgg16 cut2 | Vgg 16 cut2 Pre-trained Network + Avg global pooling + FFN+Dropout + CallBacks (EarlyStopping) | 0.96 | 0.97 | Accuracy values are definitely improved with a cut to block4_pool. |

## Final Results

The best model from scratch is the **Model_v7**. <br>
The best pre-trained netework selected is the **ResNet50**.

Models have been evaluated on test set. The test dataset contains 100 images per class, consequently it contains 500 images.

**Training and Validation Accuracy**

- 1st model is Neural Network from Scratch
- 2nd model is Neural Network with Transfer Learning

<img src="images/best_results_10k.PNG" width=600 height=400>

**Classification Reports of Test Set** 

Neural Network from Scratch

| precision | Recall | f1-score | Support |
| --------- | --------- | --------- | --------- |
| Arborio | 1.000 | 0.860 | 0.925 | 100.000 |
| Basmati | 0.980 | 1.000 | 0.990 | 100.000 | 
| Ipsala | 0.943 | 1.000 | 0.971 | 100.000 |
| Jasmine | 0.916 | 0.980 | 0.947 | 100.000 |
| Karacadag | 1.000 | 0.990 | 0.995 | 100.000 |
| accuracy | 0.966 | 0.966 | 0.966 | 0.966 |
| macro avg | 0.986 | 0.966 | 0.966 | 500.000 |
| weighted avg | 0.986 | 0.966 | 0.966 | 500.000 |

Neural Network with Transfer Learning

| precision | Recall | f1-score | Support |
| --------- | --------- | --------- | --------- |
| Arborio | 0.971 | 1.000 | 0.985 | 100.000 |
| Basmati | 0.962 | 1.000 | 0.980 | 100.000 | 
| Ipsala | 1.000 | 1.000 | 1.000 | 100.000 |
| Jasmine | 1.000 | 0.950 | 0.974 | 100.000 |
| Karacadag | 1.000 | 0.980 | 0.990 | 100.000 |
| accuracy | 0.986 | 0.986 | 0.986 | 0.986 |
| macro avg | 0.986 | 0.986 | 0.986 | 500.000 |
| weighted avg | 0.986 | 0.986 | 0.986 | 500.000 |





# How to run the code

Unless otherwise specified in the notebook section all codes can be runned in Google Colaboratory platform. All notebooks all already setted to import the necessary packages and also in this way you can easily use a GPU!

# References

[1] Koklu, M., Cinar, I., & Taspinar, Y. S. (2021). Classification of rice varieties with deep learning methods. Computers and Electronics in Agriculture, 187, 106285. https://doi.org/10.1016/j.compag.2021.106285

[2] Cinar, I., & Koklu, M. (2021). Determination of Effective and Specific Physical Features of Rice Varieties by Computer Vision In Exterior Quality Inspection. Selcuk Journal of Agriculture and Food Sciences, 35(3), 229-243. https://doi.org/10.15316/SJAFS.2021.252

[3] Cinar, I., & Koklu, M. (2022). Identification of Rice Varieties Using Machine Learning Algorithms. Journal of Agricultural Sciences https://doi.org/10.15832/ankutbd.862482

[4] Cinar, I., & Koklu, M. (2019). Classification of Rice Varieties Using Artificial Intelligence Methods. International Journal of Intelligent Systems and Applications in Engineering, 7(3), 188-194. https://doi.org/10.18201/ijisae.2019355381
