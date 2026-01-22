import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
@runtime_checkable
class SupportsAbs(Protocol[T_co]):
    """
        An ABC with one abstract method __abs__ that is covariant in its return type.
        """
    __slots__ = ()

    @abc.abstractmethod
    def __abs__(self) -> T_co:
        pass