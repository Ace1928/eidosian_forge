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
def _from_label(label):
    phase = False
    if label[0] in ('-', '+'):
        phase = label[0] == '-'
        label = label[1:]
    num_qubits = len(label)
    symp = np.zeros(2 * num_qubits + 1, dtype=bool)
    xs = symp[0:num_qubits]
    zs = symp[num_qubits:2 * num_qubits]
    for i, char in enumerate(label):
        if char not in ['I', 'X', 'Y', 'Z']:
            raise QiskitError(f"Pauli string contains invalid character: {char} not in ['I', 'X', 'Y', 'Z'].")
        if char in ('X', 'Y'):
            xs[num_qubits - 1 - i] = True
        if char in ('Z', 'Y'):
            zs[num_qubits - 1 - i] = True
    symp[-1] = phase
    return symp