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
def _count_nonzero(input_tensor, dtype=dtypes.int64):
    """Same as math_ops.count_nonzero.

  The reduction is done in dtype, which can be faster for 32-bit dtypes.

  Args:
      input_tensor: numeric tensor
      dtype: reduction dtype

  Returns:
      number of nonzero values with type dtype
  """
    with ops.name_scope('count_nonzero', values=[input_tensor]):
        zero = array_ops.zeros([], dtype=input_tensor.dtype)
        nonzero_count = math_ops.reduce_sum(math_ops.cast(math_ops.not_equal(input_tensor, zero), dtype=dtype), name='nonzero_count')
        return nonzero_count