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
def _tuple_str(obj_name, fields):
    if not fields:
        return '()'
    return f'({','.join([f'{obj_name}.{f.name}' for f in fields])},)'