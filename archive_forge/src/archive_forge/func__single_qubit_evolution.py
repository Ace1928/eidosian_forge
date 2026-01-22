from typing import Callable, Optional, Union, Any, Dict
from functools import partial
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from .evolution_synthesis import EvolutionSynthesis
def _single_qubit_evolution(pauli, time):
    definition = QuantumCircuit(pauli.num_qubits)
    for i, pauli_i in enumerate(reversed(pauli.to_label())):
        if pauli_i == 'X':
            definition.rx(2 * time, i)
        elif pauli_i == 'Y':
            definition.ry(2 * time, i)
        elif pauli_i == 'Z':
            definition.rz(2 * time, i)
    return definition