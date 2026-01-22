import math
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_arg_values
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
class ConvolutionOrthogonal1D(ConvolutionOrthogonal):
    """Initializer that generates a 1D orthogonal kernel for ConvNets.

  The shape of the tensor must have length 3. The number of input
  filters must not exceed the number of output filters.
  The orthogonality(==isometry) is exact when the inputs are circular padded.
  There are finite-width effects with non-circular padding (e.g. zero padding).
  See algorithm 1 in (Xiao et al., 2018).

  Args:
    gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
      The 2-norm of an input is multiplied by a factor of `gain` after applying
      this convolution.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  """

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        if len(shape) != 3:
            raise ValueError(f'The tensor to initialize, specified by argument `shape` must be three-dimensional. Received shape={shape}')
        if shape[-2] > shape[-1]:
            raise ValueError(f'In_filters, specified by shape[-2]={shape[-2]} cannot be greater than out_filters, specified by shape[-1]={shape[-1]}.')
        kernel = self._orthogonal_kernel(shape[0], shape[-2], shape[-1])
        kernel *= math_ops.cast(self.gain, dtype=dtype)
        return kernel

    def _dict_to_tensor(self, x, k):
        """Convert a dictionary to a tensor.

    Args:
      x: A dictionary of length k.
      k: Dimension of x.

    Returns:
      A tensor with the same dimension.
    """
        return array_ops_stack.stack([x[i] for i in range(k)])

    def _block_orth(self, projection_matrix):
        """Construct a kernel.

    Used to construct orthgonal kernel.

    Args:
      projection_matrix: A symmetric projection matrix of size n x n.

    Returns:
      [projection_matrix, (1 - projection_matrix)].
    """
        n = projection_matrix.shape.as_list()[0]
        kernel = {}
        eye = linalg_ops_impl.eye(n, dtype=self.dtype)
        kernel[0] = projection_matrix
        kernel[1] = eye - projection_matrix
        return kernel

    def _matrix_conv(self, m1, m2):
        """Matrix convolution.

    Args:
      m1: A dictionary of length k, each element is a n x n matrix.
      m2: A dictionary of length l, each element is a n x n matrix.

    Returns:
      (k + l - 1)  dictionary each element is a n x n matrix.
    Raises:
      ValueError: Ff the entries of m1 and m2 are of different dimensions.
    """
        n = m1[0].shape.as_list()[0]
        if n != m2[0].shape.as_list()[0]:
            raise ValueError(f'The entries in matrices m1 and m2 must have the same dimensions. Received m1[0].shape={m1[0].shape} and m2[0].shape={m2[0].shape}.')
        k = len(m1)
        l = len(m2)
        result = {}
        size = k + l - 1
        for i in range(size):
            result[i] = array_ops.zeros([n, n], self.dtype)
            for index in range(min(k, i + 1)):
                if i - index < l:
                    result[i] += math_ops.matmul(m1[index], m2[i - index])
        return result

    def _orthogonal_kernel(self, ksize, cin, cout):
        """Construct orthogonal kernel for convolution.

    Args:
      ksize: Kernel size.
      cin: Number of input channels.
      cout: Number of output channels.

    Returns:
      An [ksize, ksize, cin, cout] orthogonal kernel.
    Raises:
      ValueError: If cin > cout.
    """
        if cin > cout:
            raise ValueError(f'The number of input channels (cin={cin}) cannot exceed the number of output channels (cout={cout}).')
        orth = self._orthogonal_matrix(cout)[0:cin, :]
        if ksize == 1:
            return array_ops.expand_dims(orth, 0)
        p = self._block_orth(self._symmetric_projection(cout))
        for _ in range(ksize - 2):
            temp = self._block_orth(self._symmetric_projection(cout))
            p = self._matrix_conv(p, temp)
        for i in range(ksize):
            p[i] = math_ops.matmul(orth, p[i])
        return self._dict_to_tensor(p, ksize)