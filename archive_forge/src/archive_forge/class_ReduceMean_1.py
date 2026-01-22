import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceMean_1(OpRunReduceNumpy):

    def _run(self, data, axes=None, keepdims=None):
        axes = tuple(axes) if axes is not None else None
        res = np.mean(data, axis=axes, keepdims=keepdims, dtype=data.dtype)
        if keepdims == 0 and (not isinstance(res, np.ndarray)):
            res = np.array(res)
        return (res,)