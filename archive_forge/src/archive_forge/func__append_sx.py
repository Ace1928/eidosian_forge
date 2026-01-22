from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
def _append_sx(clifford, qubit):
    """Apply an SX gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= ~x & z
    x ^= z
    return clifford