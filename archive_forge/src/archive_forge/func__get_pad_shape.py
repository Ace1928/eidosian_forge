from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_dispatch  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.ops.ragged import ragged_where_op
def _get_pad_shape(params, indices, row_splits_dtype):
    """Gets the RaggedTensorDynamicShape for the pad tensor."""
    num_batch_dimensions = indices.shape.ndims - 1
    params_shape = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(params, dim_size_dtype=row_splits_dtype)
    if params.shape.ndims == indices.shape.ndims:
        if params_shape.num_inner_dimensions == 0:
            pad_dims = params_shape.partitioned_dim_sizes[:-1] + (array_ops.ones_like(params_shape.partitioned_dim_sizes[-1]),)
            return ragged_tensor_shape.RaggedTensorDynamicShape(pad_dims, [])
        else:
            return ragged_tensor_shape.RaggedTensorDynamicShape(params_shape.partitioned_dim_sizes, array_ops.concat([params_shape.inner_dim_sizes[:-1], [1]], axis=0))
    else:
        pad_dims = None
        if num_batch_dimensions == 0:
            pad_dims = (constant_op.constant(1, dtype=row_splits_dtype),) + (constant_op.constant([1], dtype=row_splits_dtype),) * (params_shape.num_partitioned_dimensions - num_batch_dimensions - 1)
        else:
            batch_dimensions = params_shape.partitioned_dim_sizes[:num_batch_dimensions]
            gather_dimension = params_shape.partitioned_dim_sizes[num_batch_dimensions]
            pad_dims = batch_dimensions + (array_ops.ones_like(gather_dimension),) * (params_shape.num_partitioned_dimensions - num_batch_dimensions)
        return ragged_tensor_shape.RaggedTensorDynamicShape(pad_dims, params_shape.inner_dim_sizes)