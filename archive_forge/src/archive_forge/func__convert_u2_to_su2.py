from __future__ import annotations
from collections.abc import Sequence
import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
def _convert_u2_to_su2(u2_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a U(2) matrix to SU(2) by adding a global phase."""
    z = 1 / np.sqrt(np.linalg.det(u2_matrix))
    su2_matrix = z * u2_matrix
    phase = np.arctan2(np.imag(z), np.real(z))
    return (su2_matrix, phase)