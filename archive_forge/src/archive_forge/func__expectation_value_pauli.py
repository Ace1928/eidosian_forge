from __future__ import annotations
import copy
import re
from numbers import Number
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.operator import Operator, BaseOperator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit._accelerate.pauli_expval import (
def _expectation_value_pauli(self, pauli, qargs=None):
    """Compute the expectation value of a Pauli.

        Args:
            pauli (Pauli): a Pauli operator to evaluate expval of.
            qargs (None or list): subsystems to apply operator on.

        Returns:
            complex: the expectation value.
        """
    n_pauli = len(pauli)
    if qargs is None:
        qubits = np.arange(n_pauli)
    else:
        qubits = np.array(qargs)
    x_mask = np.dot(1 << qubits, pauli.x)
    z_mask = np.dot(1 << qubits, pauli.z)
    pauli_phase = (-1j) ** pauli.phase if pauli.phase else 1
    if x_mask + z_mask == 0:
        return pauli_phase * np.linalg.norm(self.data)
    if x_mask == 0:
        return pauli_phase * expval_pauli_no_x(self.data, self.num_qubits, z_mask)
    x_max = qubits[pauli.x][-1]
    y_phase = (-1j) ** pauli._count_y()
    y_phase = y_phase[0]
    return pauli_phase * expval_pauli_with_x(self.data, self.num_qubits, z_mask, x_mask, y_phase, x_max)