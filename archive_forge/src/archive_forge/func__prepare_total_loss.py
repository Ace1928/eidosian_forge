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
def _prepare_total_loss(self, masks):
    """Computes total loss from loss functions.

    Args:
        masks: List of mask values corresponding to each model output.

    Returns:
        A list of loss weights of python floats.

    Raises:
        TypeError: If model run_eagerly is True.
    """
    if self.run_eagerly:
        raise TypeError('total loss can not be computed when compiled with run_eagerly = True.')
    loss_list = []
    with backend.name_scope('loss'):
        for endpoint, mask in zip(self._training_endpoints, masks):
            if endpoint.should_skip_target():
                continue
            y_true = endpoint.training_target.target
            y_pred = endpoint.output
            loss_fn = endpoint.loss_fn
            loss_weight = endpoint.loss_weight
            loss_name = endpoint.loss_name()
            sample_weight = endpoint.sample_weight
            with backend.name_scope(loss_name):
                if mask is not None:
                    mask = math_ops.cast(mask, y_pred.dtype)
                    if sample_weight is None:
                        sample_weight = mask
                    else:
                        mask, _, sample_weight = losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=sample_weight)
                        sample_weight *= mask
                if hasattr(loss_fn, 'reduction'):
                    per_sample_losses = loss_fn.call(y_true, y_pred)
                    weighted_losses = losses_utils.compute_weighted_loss(per_sample_losses, sample_weight=sample_weight, reduction=losses_utils.ReductionV2.NONE)
                    loss_reduction = loss_fn.reduction
                    if loss_reduction == losses_utils.ReductionV2.AUTO:
                        loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
                    output_loss = losses_utils.reduce_weighted_loss(weighted_losses, reduction=loss_reduction)
                else:
                    output_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
                    loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
            if len(self.outputs) > 1:
                endpoint.output_loss_metric(output_loss)
            if loss_reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
                output_loss = losses_utils.scale_loss_for_distribution(output_loss)
            loss_list.append(loss_weight * output_loss)
        if not loss_list and (not self.losses):
            raise ValueError('The model cannot be compiled because it has no loss to optimize.')
        custom_losses = self.get_losses_for(None) + self.get_losses_for(self.inputs)
        if custom_losses:
            total_custom_loss = math_ops.add_n(losses_utils.cast_losses_to_common_dtype(custom_losses))
            loss_list.append(losses_utils.scale_loss_for_distribution(total_custom_loss))
        loss_list = losses_utils.cast_losses_to_common_dtype(loss_list)
        if loss_list:
            total_loss = math_ops.add_n(loss_list)
        else:
            total_loss = 0.0
    return total_loss