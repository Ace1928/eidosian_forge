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
def _broadcast_batch_dims(self, x, spectrum):
    """Broadcast batch dims of batch matrix `x` and spectrum."""
    spectrum = tensor_conversion.convert_to_tensor_v2_with_dispatch(spectrum, name='spectrum')
    batch_shape = self._batch_shape_tensor(shape=self._shape_tensor(spectrum=spectrum))
    spec_mat = array_ops.reshape(spectrum, array_ops.concat((batch_shape, [-1, 1]), axis=0))
    x, spec_mat = linear_operator_util.broadcast_matrix_batch_dims((x, spec_mat))
    x_batch_shape = array_ops.shape(x)[:-2]
    spectrum_shape = array_ops.shape(spectrum)
    spectrum = array_ops.reshape(spec_mat, array_ops.concat((x_batch_shape, self._block_shape_tensor(spectrum_shape=spectrum_shape)), axis=0))
    return (x, spectrum)