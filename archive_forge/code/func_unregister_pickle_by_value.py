import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
def unregister_pickle_by_value(module):
    """Unregister that the input module should be pickled by value."""
    if not isinstance(module, types.ModuleType):
        raise ValueError(f'Input should be a module object, got {str(module)} instead')
    if module.__name__ not in _PICKLE_BY_VALUE_MODULES:
        raise ValueError(f'{module} is not registered for pickle by value')
    else:
        _PICKLE_BY_VALUE_MODULES.remove(module.__name__)