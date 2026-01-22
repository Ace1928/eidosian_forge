import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
@functools.total_ordering
class _QubitAsQid(Qid):

    def __init__(self, qubit: Qid, dimension: int):
        self._qubit = qubit
        self._dimension = dimension
        self.validate_dimension(dimension)

    @property
    def qubit(self) -> Qid:
        return self._qubit

    @property
    def dimension(self) -> int:
        return self._dimension

    def with_dimension(self, dimension: int) -> Qid:
        """Returns a copy with a different dimension or number of levels."""
        return self.qubit.with_dimension(dimension)

    def _comparison_key(self) -> Any:
        return self._qubit._cmp_tuple()[:-1]

    def __repr__(self) -> str:
        return f'{self.qubit!r}.with_dimension({self.dimension})'

    def __str__(self) -> str:
        return f'{self.qubit!s} (d={self.dimension})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['qubit', 'dimension'])