from typing import Tuple
import numpy as np
from onnx.reference.op_run import OpRun
def _gather_nd_impl(data: np.ndarray, indices: np.ndarray, batch_dims: int) -> Tuple[np.ndarray]:
    data_rank = len(data.shape)
    batch_dims_shape = []
    batch_dims_size = 1
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]
    output_shape = batch_dims_shape + list(indices.shape)[batch_dims:-1] if indices.shape[-1] == data_rank - batch_dims else batch_dims_shape + list(indices.shape)[batch_dims:-1] + list(data.shape)[batch_dims + indices.shape[-1]:]
    output_data_buffer = []
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])
    reshaped_data = data.reshape((batch_dims_size,) + data.shape[batch_dims:])
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[batch_dim, *gather_index])
    return (np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape),)