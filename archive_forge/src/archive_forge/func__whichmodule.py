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
def _whichmodule(obj, name):
    """Find the module an object belongs to.

    This function differs from ``pickle.whichmodule`` in two ways:
    - it does not mangle the cases where obj's module is __main__ and obj was
      not found in any module.
    - Errors arising during module introspection are ignored, as those errors
      are considered unwanted side effects.
    """
    if sys.version_info[:2] < (3, 7) and isinstance(obj, typing.TypeVar):
        if name is not None and getattr(typing, name, None) is obj:
            return 'typing'
        else:
            module_name = None
    else:
        module_name = getattr(obj, '__module__', None)
    if module_name is not None:
        return module_name
    for module_name, module in sys.modules.copy().items():
        if module_name == '__main__' or module is None or (not isinstance(module, types.ModuleType)):
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except Exception:
            pass
    return None