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
class Qid(metaclass=abc.ABCMeta):
    """Identifies a quantum object such as a qubit, qudit, resonator, etc.

    Child classes represent specific types of objects, such as a qubit at a
    particular location on a chip or a qubit with a particular name.

    The main criteria that a custom qid must satisfy is *comparability*. Child
    classes meet this criteria by implementing the `_comparison_key` method. For
    example, `cirq.LineQubit`'s `_comparison_key` method returns `self.x`. This
    ensures that line qubits with the same `x` are equal, and that line qubits
    will be sorted ascending by `x`. `Qid` implements all equality,
    comparison, and hashing methods via `_comparison_key`.
    """

    @abc.abstractmethod
    def _comparison_key(self) -> Any:
        """Returns a value used to sort and compare this qubit with others.

        By default, qubits of differing type are sorted ascending according to
        their type name. Qubits of the same type are then sorted using their
        comparison key.
        """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns the dimension or the number of quantum levels this qid has.
        E.g. 2 for a qubit, 3 for a qutrit, etc.
        """

    @staticmethod
    def validate_dimension(dimension: int) -> None:
        """Raises an exception if `dimension` is not positive.

        Raises:
            ValueError: `dimension` is not positive.
        """
        if dimension < 1:
            raise ValueError(f'Wrong qid dimension. Expected a positive integer but got {dimension}.')

    def with_dimension(self, dimension: int) -> 'Qid':
        """Returns a new qid with a different dimension.

        Child classes can override.  Wraps the qubit object by default.

        Args:
            dimension: The new dimension or number of levels.
        """
        if dimension == self.dimension:
            return self
        return _QubitAsQid(self, dimension=dimension)

    def _cmp_tuple(self):
        return (type(self).__name__, repr(type(self)), self._comparison_key(), self.dimension)

    @cached_method
    def __hash__(self) -> int:
        return hash((Qid, self._comparison_key()))

    def __eq__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() == other._cmp_tuple()

    def __ne__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() != other._cmp_tuple()

    def __lt__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() < other._cmp_tuple()

    def __gt__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() > other._cmp_tuple()

    def __le__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() <= other._cmp_tuple()

    def __ge__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() >= other._cmp_tuple()

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        """Circuit symbol for qids defaults to the string representation."""
        return protocols.CircuitDiagramInfo(wire_symbols=(str(self),))