from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
def _checked_type_create(cls, source_data, _factory_fields=None, ignore_extra=False):
    if isinstance(source_data, cls):
        return source_data
    types = get_types(cls._checked_types)
    checked_type = next((t for t in types if issubclass(t, CheckedType)), None)
    if checked_type:
        return cls([checked_type.create(data, ignore_extra=ignore_extra) if not any((isinstance(data, t) for t in types)) else data for data in source_data])
    return cls(source_data)