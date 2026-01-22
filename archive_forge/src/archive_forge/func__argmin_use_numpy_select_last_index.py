import numpy as np
from onnx.reference.op_run import OpRun
def _argmin_use_numpy_select_last_index(data, axis=0, keepdims=True):
    data = np.flip(data, axis)
    result = np.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = np.expand_dims(result, axis)
    return result.astype(np.int64)