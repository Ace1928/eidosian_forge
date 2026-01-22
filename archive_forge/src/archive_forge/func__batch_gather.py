from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
def _batch_gather(params, indices, axis, batch_dims):
    """Helper that implements the body for ragged gather() when batch_dims>0.

  Args:
    params: The tensor from which to gather values.
    indices: The indices of values to gather.
    axis: The axis in `params` to gather `indices` from.
    batch_dims: The number of batch dimensions.

  Returns:
    A potentially ragged tensor.
  """
    if not params.shape[:batch_dims].is_compatible_with(indices.shape[:batch_dims]):
        raise ValueError('batch shape from indices %s does not match params shape %s' % (indices.shape[:batch_dims], params.shape))
    if batch_dims > 1:
        if not isinstance(params, ragged_tensor.RaggedTensor):
            if indices.uniform_row_length is None:
                raise ValueError('batch shape from indices does not match params shape: ragged indices dimension corresponds to uniform params dimension')
            params = ragged_tensor.RaggedTensor.from_tensor(params, ragged_rank=1, row_splits_dtype=indices.row_splits.dtype)
        if not isinstance(indices, ragged_tensor.RaggedTensor):
            if params.uniform_row_length is None:
                raise ValueError('batch shape from indices does not match params shape: ragged params dimension corresponds to uniform indices dimension')
            indices = ragged_tensor.RaggedTensor.from_tensor(indices, ragged_rank=1, row_splits_dtype=params.row_splits.dtype)
        return params.with_values(_gather(params.values, indices.values, axis - 1, batch_dims - 1))
    if axis > 1:
        if not isinstance(indices, ragged_tensor.RaggedTensor):
            adjusted_indices = params.with_values(array_ops.repeat(indices, params.row_lengths(), 0))
        else:
            if not isinstance(params, ragged_tensor.RaggedTensor):
                params = ragged_tensor.RaggedTensor.from_tensor(params, ragged_rank=1, row_splits_dtype=indices.row_splits.dtype)
            adjusted_indices = _gather(indices, params.with_values(array_ops.repeat(math_ops.range(params.nrows()), params.row_lengths())), 0, 0)
        return _batch_gather(params, adjusted_indices, axis, batch_dims + 1)
    if indices.shape.rank is None:
        raise ValueError('rank(indices) must be known statically')
    assert batch_dims == 1
    flat_params = _flatten_dims_0_and_1(params)
    adjustments = _row_starts(params, indices.dtype)
    adjustments = _increase_rank_to(adjustments, indices.shape.ndims)
    adjusted_indices = indices + adjustments
    return _gather(flat_params, adjusted_indices, axis - 1, 0)