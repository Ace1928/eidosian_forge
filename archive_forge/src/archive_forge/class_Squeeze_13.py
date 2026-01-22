import numpy as np
from onnx.reference.op_run import OpRun
class Squeeze_13(OpRun):

    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        self.axes = None

    def _run(self, data, axes=None):
        if axes is not None:
            if hasattr(axes, '__iter__'):
                sq = np.squeeze(data, axis=tuple(axes))
            else:
                sq = np.squeeze(data, axis=axes)
        else:
            sq = np.squeeze(data)
        return (sq,)