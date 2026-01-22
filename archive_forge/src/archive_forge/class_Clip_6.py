import numpy as np
from onnx.reference.op_run import OpRun
class Clip_6(OpRun):

    def _run(self, data, min=None, max=None):
        amin = min
        amax = max
        res = data if amin is amax is None else np.clip(data, amin, amax)
        return (res,) if res.dtype == data.dtype else (res.astype(data.dtype),)