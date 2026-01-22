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
def _merge_UCGate_and_diag(single_qubit_gates, diag):
    for i, gate in enumerate(single_qubit_gates):
        single_qubit_gates[i] = np.array([[diag[2 * i], 0.0], [0.0, diag[2 * i + 1]]]).dot(gate)
    return single_qubit_gates