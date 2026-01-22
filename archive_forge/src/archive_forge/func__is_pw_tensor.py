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
@_is_pauli_word.register
def _is_pw_tensor(observable: Tensor):
    pauli_word_names = ['Identity', 'PauliX', 'PauliY', 'PauliZ']
    return set(observable.name).issubset(pauli_word_names)