import copy
from typing import Callable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from . import calc_inverse_matrix, check_invertible_binary_matrix
Check that the synthesized circuit qc fits linear nearest neighbor connectivity.

    Args:
        qc: a :class:`.QuantumCircuit` containing only CX and single qubit gates.

    Returns:
        bool: True if the circuit has linear nearest neighbor connectivity.

    Raises:
        CircuitError: if qc has a non-CX two-qubit gate.
    