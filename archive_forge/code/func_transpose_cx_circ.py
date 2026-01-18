import copy
from typing import Callable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from . import calc_inverse_matrix, check_invertible_binary_matrix
def transpose_cx_circ(qc: QuantumCircuit):
    """Takes a circuit having only CX gates, and calculates its transpose.
    This is done by recursively replacing CX(i, j) with CX(j, i) in all instructions.

    Args:
        qc: a :class:`.QuantumCircuit` containing only CX gates.

    Returns:
        QuantumCircuit: the transposed circuit.

    Raises:
        CircuitError: if qc has a non-CX gate.
    """
    transposed_circ = QuantumCircuit(qc.qubits, qc.clbits, name=qc.name + '_transpose')
    for instruction in reversed(qc.data):
        if instruction.operation.name != 'cx':
            raise CircuitError('The circuit contains non-CX gates.')
        transposed_circ._append(instruction.replace(qubits=reversed(instruction.qubits)))
    return transposed_circ