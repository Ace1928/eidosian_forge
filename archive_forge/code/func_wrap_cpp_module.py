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
def wrap_cpp_module(cpp_module):
    """Wrap this torch._C.ScriptModule in a Python ScriptModule, recursively for all submodules."""

    def init_fn(script_module):
        for name, cpp_module in torch._C.ModuleDict(script_module._c).items():
            setattr(script_module, name, wrap_cpp_module(cpp_module))
        script_module._concrete_type = torch._C.ConcreteModuleType.from_jit_type(script_module._c._type())
        for idx, fn in enumerate(script_module._c._get_forward_pre_hooks()):
            script_module._forward_pre_hooks[idx] = fn
        for idx, fn in enumerate(script_module._c._get_forward_hooks()):
            script_module._forward_hooks[idx] = fn
    return torch.jit.RecursiveScriptModule._construct(cpp_module, init_fn)