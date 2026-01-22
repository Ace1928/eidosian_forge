from typing import (
import abc
import collections
import itertools
import sympy
from cirq import protocols
from cirq._doc import document
from cirq.study import resolver
class SingleSweep(Sweep):
    """A simple sweep over one parameter with values from an iterator."""

    def __init__(self, key: 'cirq.TParamKey') -> None:
        if isinstance(key, sympy.Symbol):
            key = str(key)
        self.key = key

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._tuple() == other._tuple()

    def __hash__(self) -> int:
        return hash((self.__class__, self._tuple()))

    @abc.abstractmethod
    def _tuple(self) -> Tuple[Any, ...]:
        pass

    @property
    def keys(self) -> List['cirq.TParamKey']:
        return [self.key]

    def param_tuples(self) -> Iterator[Params]:
        for value in self._values():
            yield ((self.key, value),)

    @abc.abstractmethod
    def _values(self) -> Iterator[float]:
        pass