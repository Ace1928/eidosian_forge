import numpy as np
from onnx.reference.ops._op import OpRunReduceNumpy
class ReduceLogSumExp_18(OpRunReduceNumpy):

    def _run(self, data, axes=None, keepdims=1, noop_with_empty_axes=0):
        if self.is_axes_empty(axes) and noop_with_empty_axes:
            return (data,)
        axes = self.handle_axes(axes)
        keepdims = keepdims != 0
        if data.size == 0:
            return self.reduce_constant(data, -np.inf, axes, keepdims)
        return compute_log_sum_exp(data, axes, keepdims)