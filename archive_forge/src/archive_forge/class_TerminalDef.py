from abc import abstractmethod, ABC
import re
from contextlib import suppress
from typing import (
from types import ModuleType
import warnings
from .utils import classify, get_regexp_width, Serialize, logger
from .exceptions import UnexpectedCharacters, LexError, UnexpectedToken
from .grammar import TOKEN_DEFAULT_PRIORITY
from copy import copy
class TerminalDef(Serialize):
    """A definition of a terminal"""
    __serialize_fields__ = ('name', 'pattern', 'priority')
    __serialize_namespace__ = (PatternStr, PatternRE)
    name: str
    pattern: Pattern
    priority: int

    def __init__(self, name: str, pattern: Pattern, priority: int=TOKEN_DEFAULT_PRIORITY) -> None:
        assert isinstance(pattern, Pattern), pattern
        self.name = name
        self.pattern = pattern
        self.priority = priority

    def __repr__(self):
        return '%s(%r, %r)' % (type(self).__name__, self.name, self.pattern)

    def user_repr(self) -> str:
        if self.name.startswith('__'):
            return self.pattern.raw or self.name
        else:
            return self.name