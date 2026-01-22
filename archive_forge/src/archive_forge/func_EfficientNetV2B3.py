import copy
import math
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.applications.efficientnet_v2.EfficientNetV2B3', 'keras.applications.EfficientNetV2B3')
def EfficientNetV2B3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax', include_preprocessing=True):
    return EfficientNetV2(width_coefficient=1.2, depth_coefficient=1.4, default_size=300, model_name='efficientnetv2-b3', include_top=include_top, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling, classes=classes, classifier_activation=classifier_activation, include_preprocessing=include_preprocessing)