from __future__ import annotations
import functools
import itertools
import re
from typing import Literal
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import _count_y
from .base_pauli import BasePauli
from .clifford_circuits import _append_circuit, _append_operation
def _pad_with_identity(self, clifford, qargs):
    """Pad Clifford with identities on other subsystems."""
    if qargs is None:
        return clifford
    padded = Clifford(np.eye(2 * self.num_qubits, dtype=bool), validate=False, copy=False)
    inds = list(qargs) + [self.num_qubits + i for i in qargs]
    for i, pos in enumerate(qargs):
        padded.tableau[inds, pos] = clifford.tableau[:, i]
        padded.tableau[inds, self.num_qubits + pos] = clifford.tableau[:, clifford.num_qubits + i]
    padded.phase[inds] = clifford.phase
    return padded