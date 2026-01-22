from __future__ import annotations
import numpy as np
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
def _append_ecr(clifford, qubit0, qubit1):
    """Apply an ECR gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _append_s(clifford, qubit0)
    clifford = _append_sx(clifford, qubit1)
    clifford = _append_cx(clifford, qubit0, qubit1)
    clifford = _append_x(clifford, qubit0)
    return clifford