import numpy as np
from onnx.reference.op_run import OpRun
class Squeeze_1(OpRun):

    def _run(self, data, axes=None):
        if isinstance(axes, np.ndarray):
            axes = tuple(axes)
        elif axes in [[], ()]:
            axes = None
        elif isinstance(axes, list):
            axes = tuple(axes)
        if isinstance(axes, (tuple, list)):
            sq = data
            for a in reversed(axes):
                sq = np.squeeze(sq, axis=a)
        else:
            sq = np.squeeze(data, axis=axes)
        return (sq,)