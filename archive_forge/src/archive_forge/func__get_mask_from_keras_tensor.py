import copy
import tensorflow.compat.v2 as tf
from keras.src import layers as layer_module
from keras.src.engine import base_layer
from keras.src.engine import functional
from keras.src.engine import input_layer
from keras.src.engine import training
from keras.src.engine import training_utils
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization as legacy_serialization
from keras.src.saving.legacy.saved_model import model_serialization
from keras.src.utils import generic_utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from keras.src.utils import traceback_utils
from tensorflow.python.util.tf_export import keras_export
def _get_mask_from_keras_tensor(kt):
    return getattr(kt, '_keras_mask', None)