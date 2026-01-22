from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import check_invertible_binary_matrix
from qiskit.circuit.library.generalized_gates.permutation import PermutationGate
from qiskit.quantum_info import Clifford
@staticmethod
def _permutation_to_mat(perm):
    """This creates a nxn matrix from a given permutation gate."""
    nq = len(perm.pattern)
    mat = np.zeros((nq, nq), dtype=bool)
    for i, j in enumerate(perm.pattern):
        mat[i, j] = True
    return mat