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
def _apply_ucg(m, k, single_qubit_gates):
    num_qubits = int(np.log2(m.shape[0]))
    num_col = m.shape[1]
    spacing = 2 ** (num_qubits - k - 1)
    for j in range(2 ** (num_qubits - 1)):
        i = j // spacing * spacing + j
        gate_index = i // 2 ** (num_qubits - k)
        for col in range(num_col):
            m[np.array([i, i + spacing]), np.array([col, col])] = np.ndarray.flatten(single_qubit_gates[gate_index].dot(np.array([[m[i, col]], [m[i + spacing, col]]]))).tolist()
    return m