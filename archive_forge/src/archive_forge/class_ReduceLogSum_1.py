import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceLogSum_1(OpRunReduceNumpy):

    def _run(self, data, axes=None, keepdims=True):
        tax = tuple(axes) if axes is not None else None
        if data.size == 0:
            return self.reduce_constant(data, -np.inf, tax, keepdims)
        res = np.sum(data, axis=tax, keepdims=keepdims)
        if len(res.shape) > 0:
            return (np.log(res, out=res),)
        return (np.log(res),)