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
def _apply_diagonal_gate(m, action_qubit_labels, diag):
    num_qubits = int(np.log2(m.shape[0]))
    num_cols = m.shape[1]
    basis_states = list(itertools.product([0, 1], repeat=num_qubits))
    for state in basis_states:
        state_on_action_qubits = [state[i] for i in action_qubit_labels]
        diag_index = _bin_to_int(state_on_action_qubits)
        i = _bin_to_int(state)
        for j in range(num_cols):
            m[i, j] = diag[diag_index] * m[i, j]
    return m