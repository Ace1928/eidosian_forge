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
def try_compile_fn(fn, loc):
    if _jit_internal.is_ignored_fn(fn):
        return None
    if isinstance(fn, torch.nn.Module):
        return None
    if not inspect.isfunction(fn) and (not inspect.ismethod(fn)):
        raise RuntimeError(f'`{fn}` is not a function. Recursive scripting only supports Python functions or methods currently.\nConsider manually annotating `{fn}` with @torch.jit.script.')
    rcb = _jit_internal.createResolutionCallbackFromClosure(fn)
    return torch.jit.script(fn, _rcb=rcb)