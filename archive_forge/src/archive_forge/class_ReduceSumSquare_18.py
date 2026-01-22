import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceSumSquare_18(OpRunReduceNumpy):

    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        if self.is_axes_empty(axes) and noop_with_empty_axes != 0:
            return (np.square(data),)
        axes = self.handle_axes(axes)
        keepdims = keepdims != 0
        res = np.sum(np.square(data), axis=axes, keepdims=keepdims)
        if keepdims == 0 and (not isinstance(res, np.ndarray)):
            res = np.array(res)
        return (res,)