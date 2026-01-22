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
def _get_overloaded_methods(method, mod_class):
    if not hasattr(method, '__name__'):
        return None
    qual_name = _qualified_name(method)
    class_name_map = _overloaded_methods.get(qual_name, None)
    if class_name_map is None:
        return None
    overloads = class_name_map.get(mod_class.__name__, None)
    if overloads is None:
        return None
    method_line_no = get_source_lines_and_file(method)[1]
    mod_class_fileno = get_source_lines_and_file(mod_class)[1]
    mod_end_fileno = mod_class_fileno + len(get_source_lines_and_file(mod_class)[0])
    if not (method_line_no >= mod_class_fileno and method_line_no <= mod_end_fileno):
        raise Exception('Overloads are not useable when a module is redeclared within the same file: ' + str(method))
    return overloads