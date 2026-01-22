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
def _update_sample_weight_modes(self, sample_weights=None):
    """Updates sample weight modes based on training/eval inputs.

    Sample weight placeholders will be created for all or no outputs
    based on whether sample_weight is provided for any output.

    If model contains `_sample_weight_modes` we check if the input
    `sample_weights` corresponds to the sample weight modes.
      1. Set sample weight mode to be 'temporal' for output i, if `compile`
        sample_weight_mode was set to `temporal` and sample weight inputs
        are given for one or more outputs.
      2. Set sample weight mode to be 'samplewise' for output i, if `compile`
        sample_weight_mode was not set and sample weight inputs are given for
        one or more outputs.
      3. Reset sample weight mode to None for output i if sample weight mode
        was set but there is no sample weight input.

    Args:
      sample_weights: List of sample weights of the same length as model outputs
        or None.
    """
    if not self._is_compiled:
        return
    if sample_weights and any((s is not None for s in sample_weights)):
        for endpoint in self._training_endpoints:
            endpoint.sample_weight_mode = endpoint.sample_weight_mode or 'samplewise'
    else:
        for endpoint in self._training_endpoints:
            endpoint.sample_weight_mode = None