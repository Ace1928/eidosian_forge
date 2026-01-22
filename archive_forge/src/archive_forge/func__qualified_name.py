import ast
import builtins
import collections
import contextlib
import enum
import inspect
import io
import pickle
import sys
import threading
import types
import typing
import warnings
import weakref
from textwrap import dedent
from typing import (  # noqa: F401
import torch
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future
def _qualified_name(obj, mangle_name=True) -> str:
    if hasattr(obj, '_jit_override_qualname'):
        return obj._jit_override_qualname
    if isinstance(obj, torch._C.ScriptFunction):
        return obj.qualified_name
    if getattr(obj, '__name__', None):
        name = obj.__name__
    elif isinstance(obj, enum.Enum):
        name = obj.name
    else:
        raise RuntimeError('Could not get name of python class object')
    if name == '<lambda>':
        name = '_lambda'
    module_name = obj.__module__
    if module_name == 'torch._classes':
        return obj.qualified_name
    if module_name is None:
        raise RuntimeError(f"Could not get qualified name for class '{name}': __module__ can't be None.")
    if package_mangling.is_mangled(module_name):
        module_name = module_name.replace('<', '_')
        module_name = module_name.replace('>', '_')
    if mangle_name:
        if module_name == '__main__':
            module_name = '__torch__'
        else:
            module_name = '__torch__.' + module_name
    if '.' in name:
        raise RuntimeError(f"Could not get qualified name for class '{name}': '{name}' is not a valid identifier")
    return module_name + '.' + name