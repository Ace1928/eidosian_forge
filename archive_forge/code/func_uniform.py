import numpy as onp
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.random.uniform', v1=[])
@np_utils.np_doc('random.uniform')
def uniform(low=0.0, high=1.0, size=None):
    dtype = np_utils.result_type(float)
    low = np_array_ops.asarray(low, dtype=dtype)
    high = np_array_ops.asarray(high, dtype=dtype)
    if size is None:
        size = array_ops.broadcast_dynamic_shape(low.shape, high.shape)
    return random_ops.random_uniform(shape=size, minval=low, maxval=high, dtype=dtype)