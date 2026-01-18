from typing import Any, Callable, Optional, overload, Union
from typing_extensions import Protocol
from cirq import protocols, _compat
Automatically implemented by the `cirq.value_equality` decorator.

        Can be manually implemented by setting `manual_cls` in the decorator.

        This method encodes the logic used to determine whether or not objects
        that have the same equivalence values but different types are considered
        to be equal. By default, this returns the decorated type. But there is
        an option (`distinct_child_types`) to make it return `type(self)`
        instead.

        Returns:
            Type used when determining if the receiving object is equal to
            another object.
        