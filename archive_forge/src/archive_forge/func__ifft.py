import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
def _ifft(self, x):
    """IFFT along the last self.block_depth dimensions of x.

    Args:
      x: `Tensor` with floating or complex dtype.  Should be in the form
        returned by self._vectorize_then_blockify.

    Returns:
      `Tensor` with `dtype` `complex64`.
    """
    x_complex = _to_complex(x)
    return _IFFT_OP[self.block_depth](x_complex)