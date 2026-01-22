import numpy as np
from onnx.reference.op_run import OpRun
def _scatter_nd_impl(data, indices, updates, reduction=None):
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        if reduction == 'add':
            output[tuple(indices[i])] += updates[i]
        elif reduction == 'mul':
            output[tuple(indices[i])] *= updates[i]
        elif reduction == 'max':
            output[tuple(indices[i])] = np.maximum(output[indices[i]], updates[i])
        elif reduction == 'min':
            output[tuple(indices[i])] = np.minimum(output[indices[i]], updates[i])
        else:
            output[tuple(indices[i])] = updates[i]
    return output