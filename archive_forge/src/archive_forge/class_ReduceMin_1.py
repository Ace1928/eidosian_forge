import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceMin_1(OpRunReduceNumpy):

    def _run(self, data, axes=None, keepdims=None):
        axes = tuple(axes) if axes is not None else None
        if data.size == 0:
            maxvalue = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else np.inf
            return self.reduce_constant(data, maxvalue, axes, keepdims)
        res = np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1)
        if keepdims == 0 and (not isinstance(res, np.ndarray)):
            res = np.array(res)
        return (res,)