import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum
def _leaky_relu(x: np.ndarray, alpha: float) -> np.ndarray:
    sign = (x > 0).astype(x.dtype)
    sign -= ((sign - 1) * alpha).astype(x.dtype)
    return x * sign