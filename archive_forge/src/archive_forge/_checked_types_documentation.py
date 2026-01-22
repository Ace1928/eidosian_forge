from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector

    A CheckedPMap is a PMap which allows specifying type and invariant checks.

    >>> class IntToFloatMap(CheckedPMap):
    ...     __key_type__ = int
    ...     __value_type__ = float
    ...     __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    ...
    >>> IntToFloatMap({1: 1.5, 2: 2.25})
    IntToFloatMap({1: 1.5, 2: 2.25})
    