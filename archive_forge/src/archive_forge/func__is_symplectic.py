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
def _is_symplectic(mat):
    """Return True if input is symplectic matrix."""
    dim = len(mat) // 2
    if mat.shape != (2 * dim, 2 * dim):
        return False
    one = np.eye(dim, dtype=int)
    zero = np.zeros((dim, dim), dtype=int)
    seye = np.block([[zero, one], [one, zero]])
    arr = mat.astype(int)
    return np.array_equal(np.mod(arr.T.dot(seye).dot(arr), 2), seye)