import numpy as np
from onnx.reference.ops._op import OpRunBinaryComparison
class LessOrEqual(OpRunBinaryComparison):

    def _run(self, a, b):
        return (np.less_equal(a, b),)