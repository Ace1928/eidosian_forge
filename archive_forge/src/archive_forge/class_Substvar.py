import contextlib
import errno
import re
import sys
import typing
from abc import ABC
from collections import OrderedDict
from collections.abc import MutableMapping
from types import TracebackType
from typing import Dict, Set, Optional, Union, Iterator, IO, Iterable, TYPE_CHECKING, Type
class Substvar:
    __slots__ = ['_assignment_operator', '_value']

    def __init__(self, initial_value='', assignment_operator='='):
        self._value = initial_value
        self.assignment_operator = assignment_operator

    @property
    def assignment_operator(self):
        return self._assignment_operator

    @assignment_operator.setter
    def assignment_operator(self, new_operator):
        if new_operator not in {'=', '?='}:
            raise ValueError('Operator must be one of: "=", or "?=" - got: ' + new_operator)
        self._assignment_operator = new_operator

    def add_dependency(self, dependency_clause):
        if self._value == '':
            self._value = {dependency_clause}
            return
        if isinstance(self._value, str):
            self._value = {v.strip() for v in self._value.split(',')}
        self._value.add(dependency_clause)

    def resolve(self):
        if isinstance(self._value, set):
            return ', '.join(sorted(self._value))
        return self._value

    def __eq__(self, other):
        if other is None or not isinstance(other, Substvar):
            return False
        if self.assignment_operator != other.assignment_operator:
            return False
        return self.resolve() == other.resolve()