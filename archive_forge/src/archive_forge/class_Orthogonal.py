import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix.

  Also available via the shortcut function `tf.keras.initializers.orthogonal`.

  If the shape of the tensor to initialize is two-dimensional, it is initialized
  with an orthogonal matrix obtained from the QR decomposition of a matrix of
  random numbers drawn from a normal distribution.
  If the matrix has fewer rows than columns then the output will have orthogonal
  rows. Otherwise, the output will have orthogonal columns.

  If the shape of the tensor to initialize is more than two-dimensional,
  a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
  is initialized, where `n` is the length of the shape vector.
  The matrix is subsequently reshaped to give a tensor of the desired shape.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.Orthogonal()
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.Orthogonal()
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.

  References:
      [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)
      ([pdf](https://arxiv.org/pdf/1312.6120.pdf))
  """

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized to an orthogonal matrix.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported. If not specified, `tf.keras.backend.floatx()` is used,
       which default to `float32` unless you configured it otherwise
       (via `tf.keras.backend.set_floatx(float_dtype)`)
      **kwargs: Additional keyword arguments.
    """
        _validate_kwargs(self.__class__.__name__, kwargs, support_partition=False)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        if len(shape) < 2:
            raise ValueError('The tensor to initialize must be at least two-dimensional')
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_cols, num_rows), min(num_cols, num_rows))
        a = self._random_generator.random_normal(flat_shape, dtype=dtype)
        q, r = gen_linalg_ops.qr(a, full_matrices=False)
        d = array_ops.tensor_diag_part(r)
        q *= math_ops.sign(d)
        if num_rows < num_cols:
            q = array_ops.matrix_transpose(q)
        return self.gain * array_ops.reshape(q, shape)

    def get_config(self):
        return {'gain': self.gain, 'seed': self.seed}