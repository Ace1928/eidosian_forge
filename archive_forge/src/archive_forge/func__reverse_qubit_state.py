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
def _reverse_qubit_state(state, basis_state, epsilon):
    state = np.array(state)
    r = np.linalg.norm(state)
    if r < epsilon:
        return np.eye(2, 2)
    if basis_state == 0:
        m = np.array([[np.conj(state[0]), np.conj(state[1])], [-state[1], state[0]]]) / r
    else:
        m = np.array([[-state[1], state[0]], [np.conj(state[0]), np.conj(state[1])]]) / r
    return m