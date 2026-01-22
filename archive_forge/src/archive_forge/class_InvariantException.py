from enum import Enum
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
from typing import TypeVar, Generic
from pyrsistent._pmap import PMap, pmap
from pyrsistent._pset import PSet, pset
from pyrsistent._pvector import PythonPVector, python_pvector
class InvariantException(Exception):
    """
    Exception raised from a :py:class:`CheckedType` when invariant tests fail or when a mandatory
    field is missing.

    Contains two fields of interest:
    invariant_errors, a tuple of error data for the failing invariants
    missing_fields, a tuple of strings specifying the missing names
    """

    def __init__(self, error_codes=(), missing_fields=(), *args, **kwargs):
        self.invariant_errors = tuple((e() if callable(e) else e for e in error_codes))
        self.missing_fields = missing_fields
        super(InvariantException, self).__init__(*args, **kwargs)

    def __str__(self):
        return super(InvariantException, self).__str__() + ', invariant_errors=[{invariant_errors}], missing_fields=[{missing_fields}]'.format(invariant_errors=', '.join((str(e) for e in self.invariant_errors)), missing_fields=', '.join(self.missing_fields))