import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
def _selectsubclass(self, key):
    subclasses = list(enumsubclasses(self.__rootclass__))
    for C in subclasses:
        if not isinstance(C.__view__, tuple):
            C.__view__ = (C.__view__,)
    choices = self.__matchkey__(key, subclasses)
    if not choices:
        return self.__rootclass__
    elif len(choices) == 1:
        return choices[0]
    else:
        return type('?', tuple(choices), {})