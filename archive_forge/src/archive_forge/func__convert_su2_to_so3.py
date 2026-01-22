from __future__ import annotations
from collections.abc import Sequence
import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
def _convert_su2_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Computes SO(3)-matrix from input SU(2)-matrix.

    Args:
        matrix: The SU(2)-matrix for which a corresponding SO(3)-matrix needs to be computed.

    Returns:
        The SO(3)-matrix corresponding to ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SU(2)-matrix.
    """
    _check_is_su2(matrix)
    matrix = matrix.astype(complex)
    a = np.real(matrix[0][0])
    b = np.imag(matrix[0][0])
    c = -np.real(matrix[0][1])
    d = -np.imag(matrix[0][1])
    rotation = np.array([[a ** 2 - b ** 2 - c ** 2 + d ** 2, 2 * a * b + 2 * c * d, -2 * a * c + 2 * b * d], [-2 * a * b + 2 * c * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * a * d + 2 * b * c], [2 * a * c + 2 * b * d, 2 * b * c - 2 * a * d, a ** 2 + b ** 2 - c ** 2 - d ** 2]], dtype=float)
    return rotation