import numpy as np
from onnx.reference.op_run import OpRun
class GlobalAveragePool(OpRun):

    def _run(self, x):
        return (_global_average_pool(x).astype(x.dtype),)