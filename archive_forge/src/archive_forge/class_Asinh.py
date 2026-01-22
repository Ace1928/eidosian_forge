import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum
class Asinh(OpRunUnaryNum):

    def _run(self, x):
        return (np.arcsinh(x),)