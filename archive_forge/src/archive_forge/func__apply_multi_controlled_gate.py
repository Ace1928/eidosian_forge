from __future__ import annotations
import itertools
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from .diagonal import Diagonal
from .uc import UCGate
from .mcg_up_to_diagonal import MCGupDiag
def _apply_multi_controlled_gate(m, control_labels, target_label, gate):
    num_qubits = int(np.log2(m.shape[0]))
    num_cols = m.shape[1]
    control_labels.sort()
    free_qubits = num_qubits - len(control_labels) - 1
    basis_states_free = list(itertools.product([0, 1], repeat=free_qubits))
    for state_free in basis_states_free:
        e1, e2 = _construct_basis_states(state_free, control_labels, target_label)
        for i in range(num_cols):
            m[np.array([e1, e2]), np.array([i, i])] = np.ndarray.flatten(gate.dot(np.array([[m[e1, i]], [m[e2, i]]]))).tolist()
    return m