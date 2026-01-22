from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
class CheckedValueTypeError(CheckedTypeError):
    """
    Raised when trying to set a value using a key with a type that doesn't match the declared type.

    Attributes:
    source_class -- The class of the collection
    expected_types  -- Allowed types
    actual_type -- The non matching type
    actual_value -- Value of the variable with the non matching type
    """
    pass