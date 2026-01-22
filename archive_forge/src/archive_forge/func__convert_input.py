import tree
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.utils import backend_utils
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
def _convert_input(self, x):
    if not isinstance(x, (tf.Tensor, tf.SparseTensor, tf.RaggedTensor)):
        if not isinstance(x, (list, tuple, int, float)):
            x = backend.convert_to_numpy(x)
        x = tf.convert_to_tensor(x)
    return x