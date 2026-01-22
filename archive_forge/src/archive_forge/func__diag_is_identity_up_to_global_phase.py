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
def _diag_is_identity_up_to_global_phase(diag, epsilon):
    if not np.abs(diag[0]) < epsilon:
        global_phase = 1.0 / diag[0]
    else:
        return False
    for d in diag:
        if not np.abs(global_phase * d - 1) < epsilon:
            return False
    return True