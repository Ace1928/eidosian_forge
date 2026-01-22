from functools import lru_cache, reduce, singledispatch
from itertools import product
from typing import List, Union
from warnings import warn
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.tape import OperationRecorder
from pennylane.wires import Wires
def _binary_matrix_from_pws(terms, num_qubits, wire_map=None):
    """Get a binary matrix representation from a list of PauliWords where each row corresponds to a
    Pauli term, which is represented by a concatenation of Z and X vectors.

    Args:
        terms (Iterable[~.PauliWord]): operators defining the Hamiltonian
        num_qubits (int): number of wires required to define the Hamiltonian
        wire_map (dict): dictionary containing all wire labels used in the Pauli words as keys, and
            unique integer labels as their values

    Returns:
        array[int]: binary matrix representation of the Hamiltonian of shape
        :math:`len(terms) * 2*num_qubits`

    **Example**

    >>> from pennylane.pauli import PauliWord
    >>> wire_map = {'a':0, 'b':1, 'c':2, 'd':3}
    >>> terms = [PauliWord({'a': 'Z', 'b': 'X'}),
    ...          PauliWord({'a': 'Z', 'c': 'Y'}),
    ...          PauliWord({'a': 'X', 'd': 'Y'})]
    >>> _binary_matrix_from_pws(terms, 4, wire_map=wire_map)
    array([[1, 0, 0, 0, 0, 1, 0, 0],
           [1, 0, 1, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 1, 0, 0, 1]])
    """
    if wire_map is None:
        all_wires = qml.wires.Wires.all_wires([term.wires for term in terms], sort=True)
        wire_map = {i: c for c, i in enumerate(all_wires)}
    binary_matrix = np.zeros((len(terms), 2 * num_qubits), dtype=int)
    for idx, pw in enumerate(terms):
        for wire, pauli_op in pw.items():
            if pauli_op in ['X', 'Y']:
                binary_matrix[idx][wire_map[wire] + num_qubits] = 1
            if pauli_op in ['Z', 'Y']:
                binary_matrix[idx][wire_map[wire]] = 1
    return binary_matrix