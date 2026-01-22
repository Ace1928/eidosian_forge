from __future__ import annotations
from collections import defaultdict
from typing import Literal
import numpy as np
import rustworkx as rx
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.mixins import GroupMixin, LinearMixin
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
def _noncommutation_graph(self, qubit_wise):
    """Create an edge list representing the non-commutation graph (Pauli Graph).

        An edge (i, j) is present if i and j are not commutable.

        Args:
            qubit_wise (bool): whether the commutation rule is applied to the whole operator,
                or on a per-qubit basis.

        Returns:
            list[tuple[int,int]]: A list of pairs of indices of the PauliList that are not commutable.
        """
    mat1 = np.array([op.z + 2 * op.x for op in self], dtype=np.int8)
    mat2 = mat1[:, None]
    qubit_anticommutation_mat = mat1 * mat2 * (mat1 - mat2)
    if qubit_wise:
        adjacency_mat = np.logical_or.reduce(qubit_anticommutation_mat, axis=2)
    else:
        adjacency_mat = np.logical_xor.reduce(qubit_anticommutation_mat, axis=2)
    return list(zip(*np.where(np.triu(adjacency_mat, k=1))))