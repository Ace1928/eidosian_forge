import numpy as np
from onnx.reference.op_run import OpRun
class _ArgMin(OpRun):

    def _run(self, data, axis=None, keepdims=None):
        return (_argmin(data, axis=axis, keepdims=keepdims),)