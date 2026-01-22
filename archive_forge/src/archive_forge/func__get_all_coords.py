from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _get_all_coords(data: np.ndarray) -> np.ndarray:
    return _cartesian([list(range(data.shape[i])) for i in range(len(data.shape))])