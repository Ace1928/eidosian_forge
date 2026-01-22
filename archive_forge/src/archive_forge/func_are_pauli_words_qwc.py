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
def are_pauli_words_qwc(lst_pauli_words):
    """Given a list of observables assumed to be valid Pauli observables, determine if they are pairwise
    qubit-wise commuting.

    This implementation has time complexity ~ O(m * n) for m Pauli words and n wires, where n is the
    number of distinct wire labels used to represent the Pauli words.

    Args:
        lst_pauli_words (list[Observable]): List of observables (assumed to be valid Pauli words).

    Returns:
        (bool): True if they are all qubit-wise commuting, false otherwise. If any of the provided
        observables are not valid Pauli words, false is returned.
    """
    if all((op.pauli_rep is not None for op in lst_pauli_words)):
        return _are_pauli_words_qwc_pauli_rep(lst_pauli_words)
    latest_op_name_per_wire = {}
    for op in lst_pauli_words:
        op_names = [op.name] if not isinstance(op.name, list) else op.name
        op_wires = op.wires.tolist()
        for op_name, wire in zip(op_names, op_wires):
            latest_op_name = latest_op_name_per_wire.get(wire, 'Identity')
            if latest_op_name != op_name and (op_name != 'Identity' and latest_op_name != 'Identity'):
                return False
            if op_name != 'Identity':
                latest_op_name_per_wire[wire] = op_name
    return True