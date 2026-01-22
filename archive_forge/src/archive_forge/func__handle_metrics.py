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
def _handle_metrics(self, outputs, targets=None, skip_target_masks=None, sample_weights=None, masks=None, return_weighted_metrics=False, return_weighted_and_unweighted_metrics=False):
    """Handles calling metric functions.

    Args:
      outputs: List of outputs (predictions).
      targets: List of targets.
      skip_target_masks: Optional. List of boolean for whether the corresponding
        target should be ignored or not.
      sample_weights: Optional list of sample weight arrays.
      masks: List of computed output mask values.
      return_weighted_metrics: Flag that indicates whether weighted metrics
        should be computed instead of unweighted metrics. This flag is ignored
        when `return_weighted_and_unweighted_metrics` is enabled.
      return_weighted_and_unweighted_metrics: Flag that is used to indicate
        whether both weighted and unweighted metrics should be computed. When
        this is not enabled, we use `return_weighted_metrics` param to indicate
        whether weighted or unweighted metrics should be returned.

    Returns:
      A list of metric result tensors.
    """
    skip_target_masks = skip_target_masks or [False] * len(outputs)
    metric_results = []
    with backend.name_scope('metrics'):
        for i in range(len(outputs)):
            if skip_target_masks[i]:
                continue
            output = outputs[i] if outputs else None
            target = targets[i] if targets else None
            output_mask = masks[i] if masks else None
            if return_weighted_and_unweighted_metrics or not return_weighted_metrics:
                metric_results.extend(self._handle_per_output_metrics(self._per_output_metrics[i], target, output, output_mask))
            if return_weighted_and_unweighted_metrics or return_weighted_metrics:
                metric_results.extend(self._handle_per_output_metrics(self._per_output_weighted_metrics[i], target, output, output_mask, weights=sample_weights[i] if sample_weights else None))
    return metric_results