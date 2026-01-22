from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence
def _cross_product_matrix(v: np.ndarray) -> np.ndarray:
    """Computes cross product matrix from vector.

    Args:
        v: Vector for which cross product matrix needs to be computed.

    Returns:
        The cross product matrix corresponding to vector ``v``.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])