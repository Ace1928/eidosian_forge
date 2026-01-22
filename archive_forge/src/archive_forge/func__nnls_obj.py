import numpy as np
import scipy.optimize
from .utils import MAX_MEM_BLOCK
from typing import Any, Optional, Tuple, Sequence
def _nnls_obj(x: np.ndarray, shape: Sequence[int], A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute the objective and gradient for NNLS"""
    x = x.reshape(shape)
    diff = np.einsum('mf,...ft->...mt', A, x, optimize=True) - B
    value = 1 / B.size * 0.5 * np.sum(diff ** 2)
    grad = 1 / B.size * np.einsum('mf,...mt->...ft', A, diff, optimize=True)
    return (value, grad.flatten())