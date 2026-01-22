import numpy as np
from onnx.reference.op_run import OpRun
class ScatterElements(OpRun):

    def _run(self, data, indices, updates, axis=None, reduction=None):
        res = scatter_elements(data, indices, updates, axis=axis, reduction=reduction)
        return (res,)