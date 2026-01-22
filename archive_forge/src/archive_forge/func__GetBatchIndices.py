from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
def _GetBatchIndices(params_shape, indices, batch_dims):
    """Addds the batch offsets to the given indices and returns the results."""
    batch_indices = indices
    indices_dtype = indices.dtype.base_dtype
    casted_params_shape = math_ops.cast(params_shape, indices_dtype)
    accum_dim_value = array_ops.ones((), dtype=indices_dtype)
    for dim in range(batch_dims, 0, -1):
        dim_value = casted_params_shape[dim - 1]
        accum_dim_value *= casted_params_shape[dim]
        start = array_ops.zeros((), dtype=indices_dtype)
        step = array_ops.ones((), dtype=indices_dtype)
        dim_indices = math_ops.range(start, dim_value, step)
        dim_indices *= accum_dim_value
        dim_shape = array_ops.concat([array_ops.tile([1], [dim - 1]), [dim_value], array_ops.tile([1], [array_ops.rank(indices) - dim])], axis=0)
        batch_indices += array_ops.reshape(dim_indices, dim_shape)
    return batch_indices