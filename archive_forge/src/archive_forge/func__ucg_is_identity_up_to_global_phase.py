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
def _ucg_is_identity_up_to_global_phase(single_qubit_gates, epsilon):
    if not np.abs(single_qubit_gates[0][0, 0]) < epsilon:
        global_phase = 1.0 / single_qubit_gates[0][0, 0]
    else:
        return False
    for gate in single_qubit_gates:
        if not np.allclose(global_phase * gate, np.eye(2, 2)):
            return False
    return True