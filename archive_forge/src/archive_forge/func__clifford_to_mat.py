from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import check_invertible_binary_matrix
from qiskit.circuit.library.generalized_gates.permutation import PermutationGate
from qiskit.quantum_info import Clifford
@staticmethod
def _clifford_to_mat(cliff):
    """This creates a nxn matrix corresponding to the given Clifford, when Clifford
        can be converted to a linear function. This is possible when the clifford has
        tableau of the form [[A, B], [C, D]], with B = C = 0 and D = A^{-1}^t, and zero
        phase vector. In this case, the required matrix is A^t.
        Raises an error otherwise.
        """
    if cliff.phase.any() or cliff.destab_z.any() or cliff.stab_x.any():
        raise CircuitError('The given clifford does not correspond to a linear function.')
    return np.transpose(cliff.destab_x)