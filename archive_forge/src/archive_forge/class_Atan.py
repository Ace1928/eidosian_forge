import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum
class Atan(OpRunUnaryNum):

    def _run(self, x):
        return (np.arctan(x),)