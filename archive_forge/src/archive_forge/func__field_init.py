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
def _field_init(f, frozen, globals, self_name, slots):
    default_name = f'_dflt_{f.name}'
    if f.default_factory is not MISSING:
        if f.init:
            globals[default_name] = f.default_factory
            value = f'{default_name}() if {f.name} is _HAS_DEFAULT_FACTORY else {f.name}'
        else:
            globals[default_name] = f.default_factory
            value = f'{default_name}()'
    elif f.init:
        if f.default is MISSING:
            value = f.name
        elif f.default is not MISSING:
            globals[default_name] = f.default
            value = f.name
    elif slots and f.default is not MISSING:
        globals[default_name] = f.default
        value = default_name
    else:
        return None
    if f._field_type is _FIELD_INITVAR:
        return None
    return _field_assign(frozen, f.name, value, self_name)