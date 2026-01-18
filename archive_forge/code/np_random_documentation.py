import numpy as onp
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
Returns samples from a normal distribution.

  Uses `tf.random_normal`.

  Args:
    *args: The shape of the output array.

  Returns:
    An ndarray with shape `args` and dtype `float64`.
  