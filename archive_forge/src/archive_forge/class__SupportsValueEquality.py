from typing import Any, Callable, Optional, overload, Union
from typing_extensions import Protocol
from cirq import protocols, _compat
class _SupportsValueEquality(Protocol):
    """An object decorated with the value equality decorator."""

    def _value_equality_values_(self) -> Any:
        """Returns a value or values that define the identity of this object.

        For example, a Point2D would be defined by the tuple (x, y) and so it
        would return `(x, y)` from this method.

        The decorated class is responsible for implementing this method.

        Returns:
            Values used when determining if the receiving object is equal to
            another object.
        """

    def _value_equality_approximate_values_(self) -> Any:
        """Returns value or values used for approximate equality.

        Approximate equality does element-wise comparison of iterable types; if
        decorated class is composed of a set of primitive types (or types
        supporting `SupportsApproximateEquality` protocol) then they can be
        given as an iterable.

        If this method is not defined by decorated class,
        `_value_equality_values_` is going to be used instead.

        Returns:
            Any type supported by `cirq.approx_eq()`.
        """
        return self._value_equality_values_()

    def _value_equality_values_cls_(self) -> Any:
        """Automatically implemented by the `cirq.value_equality` decorator.

        Can be manually implemented by setting `manual_cls` in the decorator.

        This method encodes the logic used to determine whether or not objects
        that have the same equivalence values but different types are considered
        to be equal. By default, this returns the decorated type. But there is
        an option (`distinct_child_types`) to make it return `type(self)`
        instead.

        Returns:
            Type used when determining if the receiving object is equal to
            another object.
        """