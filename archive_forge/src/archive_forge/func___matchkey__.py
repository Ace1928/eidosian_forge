import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
def __matchkey__(self, key, subclasses):
    if inspect.isclass(key):
        keys = inspect.getmro(key)
    else:
        keys = [key]
    for key in keys:
        result = [C for C in subclasses if key in C.__view__]
        if result:
            return result
    return []