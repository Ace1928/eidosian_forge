from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
def with_dim_size_dtype(self, dtype):
    if dtype not in (dtypes.int32, dtypes.int64):
        raise ValueError('dtype must be int32 or int64')
    if self.dim_size_dtype == dtype:
        return self
    return RaggedTensorDynamicShape([math_ops.cast(p, dtype) for p in self._partitioned_dim_sizes], math_ops.cast(self._inner_dim_sizes, dtype))