import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
def _canonicalize_permutation(permutation: Dict[int, int]) -> Dict[int, int]:
    return {i: j for i, j in permutation.items() if i != j}