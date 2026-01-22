import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.applications.densenet.DenseNet121', 'keras.applications.DenseNet121')
def DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax'):
    """Instantiates the Densenet121 architecture."""
    return DenseNet([6, 12, 24, 16], include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation)