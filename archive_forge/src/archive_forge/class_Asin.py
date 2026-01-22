import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum
class Asin(OpRunUnaryNum):

    def _run(self, x):
        return (np.arcsin(x),)