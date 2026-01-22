import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import models
from keras.src.applications import imagenet_utils
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.applications.MobileNetV3Large')
def MobileNetV3Large(input_shape=None, alpha=1.0, minimalistic=False, include_top=True, weights='imagenet', input_tensor=None, classes=1000, pooling=None, dropout_rate=0.2, classifier_activation='softmax', include_preprocessing=True):

    def stack_fn(x, kernel, activation, se_ratio):

        def depth(d):
            return _depth(d * alpha)
        x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
        x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
        x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14)
        return x
    return MobileNetV3(stack_fn, 1280, input_shape, alpha, 'large', minimalistic, include_top, weights, input_tensor, classes, pooling, dropout_rate, classifier_activation, include_preprocessing)