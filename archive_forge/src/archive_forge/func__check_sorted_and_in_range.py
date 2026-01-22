from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit
from .piecewise_linear_pauli_rotations import PiecewiseLinearPauliRotations
def _check_sorted_and_in_range(breakpoints, domain):
    if breakpoints is None:
        return
    if not np.all(np.diff(breakpoints) > 0):
        raise ValueError('Breakpoints must be unique and sorted.')
    if breakpoints[0] < domain[0] or breakpoints[-1] > domain[1]:
        raise ValueError('Breakpoints must be included in domain.')