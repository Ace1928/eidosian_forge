import numpy as np
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def _model_loss(model, inputs, targets, output_loss_metrics=None, sample_weights=None, training=False):
    """Calculates the loss for a given model.

  Args:
      model: The model on which metrics are being calculated.
      inputs: Either a dictionary of inputs to the model or a list of input
        arrays.
      targets: List of target arrays.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.
      sample_weights: Optional list of sample weight arrays.
      training: Whether the model should be run in inference or training mode.

  Returns:
     Returns the model output, total loss, loss value calculated using the
     specified loss function and masks for each output. The total loss includes
     regularization losses and applies masking and sample weighting
     to the loss value.
  """
    total_loss = 0
    kwargs = {}
    if model._expects_training_arg:
        kwargs['training'] = training
    if len(inputs) == 1 and (not isinstance(inputs, dict)):
        inputs = inputs[0]
    if any((isinstance(input_t, (np.ndarray, float, int)) for input_t in nest.flatten(inputs))):
        inputs = nest.map_structure(tensor_conversion.convert_to_tensor_v2_with_dispatch, inputs)
    outs = model(inputs, **kwargs)
    outs = nest.flatten(outs)
    if targets:
        targets = training_utils_v1.cast_if_floating_dtype_and_mismatch(targets, outs)
    if sample_weights:
        new_sample_weights = []
        for val in sample_weights:
            if val is not None:
                new_sample_weights.append(training_utils_v1.cast_if_floating_dtype(tensor_conversion.convert_to_tensor_v2_with_dispatch(val)))
            else:
                new_sample_weights.append(None)
        sample_weights = new_sample_weights
    masks = [getattr(t, '_keras_mask', None) for t in outs]
    targets = nest.flatten(targets)
    output_losses = []
    with backend.name_scope('loss'):
        loss_fns = [loss_fn for loss_fn in model.loss_functions if loss_fn is not None]
        custom_losses = model.losses
        if not loss_fns and (not custom_losses):
            if training:
                raise ValueError('The model cannot be trained because it has no loss to optimize.')
            else:
                raise ValueError('The model cannot be evaluated because it has no loss to compute.')
        for i, loss_fn in enumerate(loss_fns):
            weights = sample_weights[i] if sample_weights else None
            mask = masks[i]
            with backend.name_scope(model.output_names[i] + '_loss'):
                if mask is not None:
                    mask = math_ops.cast(mask, outs[i].dtype)
                    if weights is None:
                        weights = mask
                    else:
                        weights = math_ops.cast(weights, outs[i].dtype)
                        mask, _, weights = losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=weights)
                        weights *= mask
                if hasattr(loss_fn, 'reduction'):
                    per_sample_losses = loss_fn.call(targets[i], outs[i])
                    weighted_losses = losses_utils.compute_weighted_loss(per_sample_losses, sample_weight=weights, reduction=losses_utils.ReductionV2.NONE)
                    loss_reduction = loss_fn.reduction
                    if loss_reduction == losses_utils.ReductionV2.AUTO:
                        loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
                    output_loss = losses_utils.reduce_weighted_loss(weighted_losses, reduction=loss_reduction)
                else:
                    output_loss = loss_fn(targets[i], outs[i], sample_weight=weights)
                    loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
            if len(model.outputs) > 1:
                output_losses.append(output_loss_metrics[i](output_loss))
            if loss_reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
                output_loss = losses_utils.scale_loss_for_distribution(output_loss)
            total_loss += model._loss_weights_list[i] * output_loss
        if custom_losses:
            total_loss += losses_utils.scale_loss_for_distribution(math_ops.add_n(custom_losses))
    return (outs, total_loss, output_losses, masks)