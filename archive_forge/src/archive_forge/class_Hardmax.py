import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum
class Hardmax(OpRunUnaryNum):

    def _run(self, x, axis=None):
        axis = axis or self.axis
        x_argmax = np.argmax(x, axis=axis)
        y = np.zeros_like(x)
        np.put_along_axis(y, np.expand_dims(x_argmax, axis=axis), 1, axis=axis)
        return (y,)