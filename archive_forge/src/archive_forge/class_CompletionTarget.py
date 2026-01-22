from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
class CompletionTarget(metaclass=abc.ABCMeta):
    """Command-line argument completion target base class."""

    def __init__(self) -> None:
        self.name = ''
        self.path = ''
        self.base_path: t.Optional[str] = None
        self.modules: tuple[str, ...] = tuple()
        self.aliases: tuple[str, ...] = tuple()

    def __eq__(self, other):
        if isinstance(other, CompletionTarget):
            return self.__repr__() == other.__repr__()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.name.__lt__(other.name)

    def __gt__(self, other):
        return self.name.__gt__(other.name)

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        if self.modules:
            return '%s (%s)' % (self.name, ', '.join(self.modules))
        return self.name