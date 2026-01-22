import numpy as np
from onnx.reference.op_run import OpRun
def _gemm00(a, b, c, alpha, beta):
    o = np.dot(a, b) * alpha
    if c is not None and beta != 0:
        o += c * beta
    return o