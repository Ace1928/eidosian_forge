import math
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export('nn.weighted_moments', v1=[])
@dispatch.add_dispatch_support
def weighted_moments_v2(x, axes, frequency_weights, keepdims=False, name=None):
    """Returns the frequency-weighted mean and variance of `x`.

  Args:
    x: A tensor.
    axes: 1-d tensor of int32 values; these are the axes along which
      to compute mean and variance.
    frequency_weights: A tensor of positive weights which can be
      broadcast with x.
    keepdims: Produce moments with the same dimensionality as the input.
    name: Name used to scope the operation.

  Returns:
    Two tensors: `weighted_mean` and `weighted_variance`.
  """
    return weighted_moments(x=x, axes=axes, frequency_weights=frequency_weights, name=name, keep_dims=keepdims)