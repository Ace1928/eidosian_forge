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
def _evolve_s(base_pauli, qubit):
    """Update P -> S.P.Sdg"""
    x = base_pauli._x[:, qubit]
    base_pauli._z[:, qubit] ^= x
    base_pauli._phase += x.T.astype(base_pauli._phase.dtype)
    return base_pauli