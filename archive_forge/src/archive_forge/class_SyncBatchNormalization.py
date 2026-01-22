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
@keras_export('keras.layers.experimental.SyncBatchNormalization', v1=[])
@deprecation.deprecated_endpoints('keras.layers.experimental.SyncBatchNormalization')
class SyncBatchNormalization(BatchNormalizationBase):
    """Deprecated. Please use `tf.keras.layers.BatchNormalization` instead.

    Caution: `tf.keras.layers.experimental.SyncBatchNormalization` endpoint is
      deprecated and will be removed in a future release. Please use
      `tf.keras.layers.BatchNormalization` with parameter `synchronized`
      set to True
    """

    def __init__(self, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs):
        warning = '`tf.keras.layers.experimental.SyncBatchNormalization` endpoint is deprecated and will be removed in a future release. Please use `tf.keras.layers.BatchNormalization` with parameter `synchronized` set to True.'
        logging.log_first_n(logging.WARN, warning, 1)
        super().__init__(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer, moving_mean_initializer=moving_mean_initializer, moving_variance_initializer=moving_variance_initializer, beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer, beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, synchronized=True, **kwargs)