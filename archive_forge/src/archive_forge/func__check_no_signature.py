import collections
import functools
import inspect
import sys
import textwrap
import types
import warnings
from typing import Dict, List, Set, Type
import torch
import torch._jit_internal as _jit_internal
from torch._sources import fake_range
from torch.jit._builtins import _find_builtin
from torch.jit._check import AttributeTypeIsSupportedChecker
from torch.jit._state import _add_script_class, _get_script_class, _python_cu
from torch.jit.frontend import (
from torch.nn import Module
def _check_no_signature(func):
    signature = torch.jit.annotations.get_signature(func, None, fake_range(), inspect.ismethod(func))
    if signature is None:
        qual_name = _jit_internal._qualified_name(func)
        raise RuntimeError(f'Must explicitly add type annotations to overloaded functions: {qual_name}')