from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence
def _compute_rotation_axis(matrix: np.ndarray) -> np.ndarray:
    """Computes rotation axis of SO(3)-matrix.

    Args:
        matrix: The SO(3)-matrix for which rotation angle needs to be computed.

    Returns:
        The rotation axis of the SO(3)-matrix ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    _check_is_so3(matrix)
    trace = _compute_trace_so3(matrix)
    theta = math.acos(0.5 * (trace - 1))
    if math.sin(theta) > 1e-10:
        x = 1 / (2 * math.sin(theta)) * (matrix[2][1] - matrix[1][2])
        y = 1 / (2 * math.sin(theta)) * (matrix[0][2] - matrix[2][0])
        z = 1 / (2 * math.sin(theta)) * (matrix[1][0] - matrix[0][1])
    else:
        x = 1.0
        y = 0.0
        z = 0.0
    return np.array([x, y, z])