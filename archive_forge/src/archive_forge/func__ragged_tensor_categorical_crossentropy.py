import abc
import functools
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.tools.docs import doc_controls
@dispatch.dispatch_for_types(categorical_crossentropy, ragged_tensor.RaggedTensor)
def _ragged_tensor_categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0, axis=-1):
    """Implements support for handling RaggedTensors.

  Args:
    y_true: Tensor of one-hot true targets.
    y_pred: Tensor of predicted targets.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
      example, if `0.1`, use `0.1 / num_classes` for non-target labels
      and `0.9 + 0.1 / num_classes` for target labels.
    axis: The axis along which to compute crossentropy (the features axis).
        Defaults to -1.

  Returns:
    Categorical crossentropy loss value.

  Expected shape: (batch, sequence_len, n_classes) with sequence_len
  being variable per batch.
  Return shape: (batch, sequence_len).

  When used by CategoricalCrossentropy() with the default reduction
  (SUM_OVER_BATCH_SIZE), the reduction averages the loss over the
  number of elements independent of the batch. E.g. if the RaggedTensor
  has 2 batches with [2, 1] values respectivly the resulting loss is
  the sum of the individual loss values divided by 3.
  """
    fn = functools.partial(categorical_crossentropy, from_logits=from_logits, label_smoothing=label_smoothing, axis=axis)
    return _ragged_tensor_apply_loss(fn, y_true, y_pred)