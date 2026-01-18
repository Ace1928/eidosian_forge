import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
@staticmethod
def validate_permutation(permutation: Dict[int, int], n_elements: Optional[int]=None) -> None:
    if not permutation:
        return
    if set(permutation.values()) != set(permutation):
        raise IndexError('key and value sets must be the same.')
    if min(permutation) < 0:
        raise IndexError('keys of the permutation must be non-negative.')
    if n_elements is not None:
        if max(permutation) >= n_elements:
            raise IndexError('key is out of bounds.')