from typing import Union
from functools import singledispatch
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.operation import Tensor
from .utils import is_pauli_word
from .conversion import pauli_sentence
@_pauli_word_prefactor.register
def _pw_prefactor_tensor(observable: Tensor):
    if is_pauli_word(observable):
        return list(pauli_sentence(observable).values())[0]
    raise ValueError(f'Expected a valid Pauli word, got {observable}')