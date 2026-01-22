import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def _hash_fn(fields, globals):
    self_tuple = _tuple_str('self', fields)
    return _create_fn('__hash__', ('self',), [f'return hash({self_tuple})'], globals=globals)