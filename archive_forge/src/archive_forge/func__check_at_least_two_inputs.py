import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def _check_at_least_two_inputs(self, inputs):
    if not isinstance(inputs, (list, tuple)):
        raise ValueError(f'`HashedCrossing` should be called on a list or tuple of inputs. Received: inputs={inputs}')
    if len(inputs) < 2:
        raise ValueError(f'`HashedCrossing` should be called on at least two inputs. Received: inputs={inputs}')