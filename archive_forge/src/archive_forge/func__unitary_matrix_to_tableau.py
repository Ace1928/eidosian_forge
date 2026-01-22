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
@staticmethod
def _unitary_matrix_to_tableau(matrix):
    num_qubits = int(np.log2(len(matrix)))
    stab = np.empty((num_qubits, 2 * num_qubits + 1), dtype=bool)
    for i in range(num_qubits):
        label = 'I' * (num_qubits - i - 1) + 'X' + 'I' * i
        Xi = Operator.from_label(label).to_matrix()
        target = matrix @ Xi @ np.conj(matrix).T
        row = Clifford._pauli_matrix_to_row(target, num_qubits)
        if row is None:
            return None
        stab[i] = row
    destab = np.empty((num_qubits, 2 * num_qubits + 1), dtype=bool)
    for i in range(num_qubits):
        label = 'I' * (num_qubits - i - 1) + 'Z' + 'I' * i
        Zi = Operator.from_label(label).to_matrix()
        target = matrix @ Zi @ np.conj(matrix).T
        row = Clifford._pauli_matrix_to_row(target, num_qubits)
        if row is None:
            return None
        destab[i] = row
    tableau = np.vstack([stab, destab])
    return tableau