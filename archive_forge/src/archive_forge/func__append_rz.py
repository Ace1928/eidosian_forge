from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
def _append_rz(clifford, qubit, multiple):
    """Apply an Rz gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.
        multiple (int): z-rotation angle in a multiple of pi/2

    Returns:
        Clifford: the updated Clifford.
    """
    if multiple % 4 == 1:
        return _append_s(clifford, qubit)
    if multiple % 4 == 2:
        return _append_z(clifford, qubit)
    if multiple % 4 == 3:
        return _append_sdg(clifford, qubit)
    return clifford