from typing import Union
from functools import singledispatch
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.operation import Tensor
from .utils import is_pauli_word
from .conversion import pauli_sentence
@_pauli_word_prefactor.register(PauliX)
@_pauli_word_prefactor.register(PauliY)
@_pauli_word_prefactor.register(PauliZ)
@_pauli_word_prefactor.register(Identity)
def _pw_prefactor_pauli(observable: Union[PauliX, PauliY, PauliZ, Identity]):
    return 1