import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum
class Atanh(OpRunUnaryNum):

    def _run(self, x):
        return (np.arctanh(x),)