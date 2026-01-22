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
@singledispatch
def _is_pauli_word(observable):
    """
    Private implementation of is_pauli_word, to prevent all of the
    registered functions from appearing in the Sphinx docs.
    """
    return False