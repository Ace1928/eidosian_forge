from __future__ import annotations
import copy
from typing import Literal, TYPE_CHECKING
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, MultiplyMixin
def _evolve_cz(base_pauli, q1, q2):
    """Update P -> CZ.P.CZ"""
    x1 = base_pauli._x[:, q1].copy()
    x2 = base_pauli._x[:, q2].copy()
    base_pauli._z[:, q1] ^= x2
    base_pauli._z[:, q2] ^= x1
    base_pauli._phase += 2 * np.logical_and(x1, x2).T.astype(base_pauli._phase.dtype)
    return base_pauli