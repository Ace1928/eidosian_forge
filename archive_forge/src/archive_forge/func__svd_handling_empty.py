from typing import Tuple, Callable, List
import numpy as np
from cirq.linalg import combinators, predicates, tolerance
def _svd_handling_empty(mat):
    if not mat.shape[0] * mat.shape[1]:
        z = np.zeros((0, 0), dtype=mat.dtype)
        return (z, np.array([]), z)
    return np.linalg.svd(mat)