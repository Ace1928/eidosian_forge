from typing import List, Sequence, Tuple, Union, cast
import numpy as np
from pyquil.experiment._setting import TensorProductState
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.quilatom import Parameter
from pyquil.quilbase import Gate, Halt, _strip_modifiers
from pyquil.simulation.matrices import SWAP, STATES, QUANTUM_GATES
def two_swap_helper(j: int, k: int, num_qubits: int, qubit_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the permutation matrix that permutes two single-particle Hilbert
    spaces into adjacent positions.

    ALWAYS swaps j TO k. Recall that Hilbert spaces are ordered in decreasing
    qubit index order. Hence, j > k implies that j is to the left of k.

    End results:
        j == k: nothing happens
        j > k: Swap j right to k, until j at ind (k) and k at ind (k+1).
        j < k: Swap j left to k, until j at ind (k) and k at ind (k-1).

    Done in preparation for arbitrary 2-qubit gate application on ADJACENT
    qubits.

    :param j: starting qubit index
    :param k: ending qubit index
    :param num_qubits: number of qubits in Hilbert space
    :param qubit_map: current index mapping of qubits
    :return: tuple of swap matrix for the specified permutation,
             and the new qubit_map, after permutation is made
    """
    if not (0 <= j < num_qubits and 0 <= k < num_qubits):
        raise ValueError('Permutation SWAP index not valid')
    perm = np.eye(2 ** num_qubits, dtype=np.complex128)
    new_qubit_map = np.copy(qubit_map)
    if j == k:
        return (perm, new_qubit_map)
    elif j > k:
        for i in range(j, k, -1):
            perm = qubit_adjacent_lifted_gate(i - 1, SWAP, num_qubits).dot(perm)
            new_qubit_map[i - 1], new_qubit_map[i] = (new_qubit_map[i], new_qubit_map[i - 1])
    elif j < k:
        for i in range(j, k, 1):
            perm = qubit_adjacent_lifted_gate(i, SWAP, num_qubits).dot(perm)
            new_qubit_map[i], new_qubit_map[i + 1] = (new_qubit_map[i + 1], new_qubit_map[i])
    return (perm, new_qubit_map)