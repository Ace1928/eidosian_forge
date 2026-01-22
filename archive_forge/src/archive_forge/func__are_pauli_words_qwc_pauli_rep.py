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
def _are_pauli_words_qwc_pauli_rep(lst_pauli_words):
    """Given a list of observables assumed to be valid Pauli words, determine if they are pairwise
    qubit-wise commuting. This private method is used for operators that have a valid pauli
    representation"""
    basis = {}
    for op in lst_pauli_words:
        if len((pr := op.pauli_rep)) > 1:
            return False
        pw = next(iter(pr))
        for wire, pauli_type in pw.items():
            if pauli_type != 'I':
                if wire in basis and pauli_type != basis[wire]:
                    return False
                basis[wire] = pauli_type
    return True