import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('signal.vorbis_window')
@dispatch.add_dispatch_support
def vorbis_window(window_length, dtype=dtypes.float32, name=None):
    """Generate a [Vorbis power complementary window][vorbis].

  Args:
    window_length: A scalar `Tensor` indicating the window length to generate.
    dtype: The data type to produce. Must be a floating point type.
    name: An optional name for the operation.

  Returns:
    A `Tensor` of shape `[window_length]` of type `dtype`.

  [vorbis]:
    https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform#Window_functions
  """
    with ops.name_scope(name, 'vorbis_window'):
        window_length = _check_params(window_length, dtype)
        arg = math_ops.cast(math_ops.range(window_length), dtype=dtype)
        window = math_ops.sin(np.pi / 2.0 * math_ops.pow(math_ops.sin(np.pi / math_ops.cast(window_length, dtype=dtype) * (arg + 0.5)), 2.0))
    return window