from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _cubic_coeffs_antialias(ratio: float, scale: float, A: float=-0.75) -> np.ndarray:
    scale = min(scale, 1.0)

    def compute_coeff(x: float) -> float:
        x = abs(x)
        x_2 = x * x
        x_3 = x * x_2
        if x <= 1:
            return (A + 2) * x_3 - (A + 3) * x_2 + 1
        if x < 2:
            return A * x_3 - 5 * A * x_2 + 8 * A * x - 4 * A
        return 0.0
    i_start = int(np.floor(-2 / scale) + 1)
    i_end = 2 - i_start
    args = [scale * (i - ratio) for i in range(i_start, i_end)]
    coeffs = [compute_coeff(x) for x in args]
    return np.array(coeffs) / sum(coeffs)