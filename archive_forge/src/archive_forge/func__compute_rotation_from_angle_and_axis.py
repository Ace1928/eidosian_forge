from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence
def _compute_rotation_from_angle_and_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    """Computes the SO(3)-matrix corresponding to the rotation of ``angle`` about ``axis``.

    Args:
        angle: The angle of the rotation.
        axis: The axis of the rotation.

    Returns:
        SO(3)-matrix that represents a rotation of ``angle`` about ``axis``.

    Raises:
        ValueError: if ``axis`` is not a 3-dim unit vector.
    """
    if axis.shape != (3,):
        raise ValueError(f'Axis must be a 1d array of length 3, but has shape {axis.shape}.')
    if abs(np.linalg.norm(axis) - 1.0) > 0.0001:
        raise ValueError(f'Axis must have a norm of 1, but has {np.linalg.norm(axis)}.')
    res = math.cos(angle) * np.identity(3) + math.sin(angle) * _cross_product_matrix(axis)
    res += (1 - math.cos(angle)) * np.outer(axis, axis)
    return res