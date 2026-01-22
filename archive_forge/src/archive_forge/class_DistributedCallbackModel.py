import collections
import warnings
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
class DistributedCallbackModel(Model):
    """Model that is used for callbacks with tf.distribute.Strategy."""

    def __init__(self, model):
        super(DistributedCallbackModel, self).__init__()
        self.optimizer = model.optimizer

    def set_original_model(self, orig_model):
        self._original_model = orig_model

    def save_weights(self, filepath, overwrite=True, save_format=None):
        self._replicated_model.save_weights(filepath, overwrite=overwrite, save_format=save_format)

    def save(self, filepath, overwrite=True, include_optimizer=True):
        distributed_model_weights = self.get_weights()
        self._original_model.set_weights(distributed_model_weights)
        self._original_model.save(filepath, overwrite=True, include_optimizer=False)

    def load_weights(self, filepath, by_name=False):
        self._original_model.load_weights(filepath, by_name=False)
        orig_model_weights = self._original_model.get_weights()
        distributed_training_utils_v1.set_weights(self._original_model._distribution_strategy, self, orig_model_weights)

    def __getattr__(self, item):
        if item not in ('_setattr_tracking', '_layers'):
            logging.warning('You are accessing attribute ' + item + ' of the DistributedCallbackModel that may not have been set correctly.')
        return super(DistributedCallbackModel, self).__getattr__(item)