from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
def _accum_initializer(shape, dtype=dtypes.float32, partition_info=None):
    del partition_info
    return array_ops.ones(shape=shape, dtype=dtype) * self._initial_accumulator_value