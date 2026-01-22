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
def _pauli_mult(p1, p2):
    """Return the result of multiplication between two tensor products of Pauli operators.

    The Pauli operator :math:`(P_0)` is denoted by [(0, 'P')], where :math:`P` represents
    :math:`X`, :math:`Y` or :math:`Z`.

    Args:
        p1 (list[tuple[int, str]]): the first tensor product of Pauli operators
        p2 (list[tuple[int, str]]): the second tensor product of Pauli operators

    Returns
        tuple(list[tuple[int, str]], complex): list of the Pauli operators and the coefficient

    **Example**

    >>> p1 = [(0, "X"), (1, "Y")]  # X_0 @ Y_1
    >>> p2 = [(0, "X"), (2, "Y")]  # X_0 @ Y_2
    >>> _pauli_mult(p1, p2)
    ([(2, "Y"), (1, "Y")], 1.0) # p1 @ p2 = X_0 @ Y_1 @ X_0 @ Y_2
    """
    warn('_pauli_mult is deprecated. Instead, please use the PauliWord class, or regular PennyLane operators.', qml.PennyLaneDeprecationWarning)
    c = 1.0
    t1 = [t[0] for t in p1]
    t2 = [t[0] for t in p2]
    k = []
    for i in p1:
        if i[0] in t1 and i[0] not in t2:
            k.append((i[0], pauli_mult_dict[i[1]]))
        for j in p2:
            if j[0] in t2 and j[0] not in t1:
                k.append((j[0], pauli_mult_dict[j[1]]))
            if i[0] == j[0]:
                k.append((i[0], pauli_mult_dict[i[1] + j[1]]))
                if i[1] + j[1] in pauli_coeff:
                    c = c * pauli_coeff[i[1] + j[1]]
    for item in k:
        k_ = [i for i, x in enumerate(k) if x == item]
        if len(k_) >= 2:
            for j in k_[::-1][:-1]:
                del k[j]
    return (k, c)