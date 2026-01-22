import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import control_flow_util
from keras.src.utils import tf_utils
from tensorflow.python.ops.control_flow_ops import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
def _raise_for_non_sync_bn_with_renorm_and_dtensor_strategy(synchronized, training, renorm):
    if utils.running_with_dtensor_strategy() and (not synchronized) and (training == True) and renorm:
        raise NotImplementedError('Renorm for BatchNormalization under DTensor based distribution strategy is not supported at the moment. Please file a feature request if this is blocking your adoption.')