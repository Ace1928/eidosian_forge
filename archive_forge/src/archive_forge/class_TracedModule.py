import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
class TracedModule(ScriptModule):
    _disable_script_meta = True

    def __init__(self, orig, id_set=None, _compilation_unit=None):
        super().__init__()
        assert isinstance(orig, torch.nn.Module)
        id_set = set()

        class QualnameWrapper(torch.nn.Module):
            pass
        QualnameWrapper._jit_override_qualname = torch._jit_internal._qualified_name(type(orig))
        tmp_module = QualnameWrapper()

        def check_unique(param):
            if param in id_set:
                raise ValueError("TracedModules don't support parameter sharing between modules")
            id_set.add(param)
        tmp_module.training = orig.training
        for name, param in orig._parameters.items():
            if param is not None:
                tmp_module._parameters[name] = param
                check_unique(param)
        for name, buf in orig._buffers.items():
            if buf is not None:
                tmp_module._buffers[name] = buf
                check_unique(buf)
        for name, val in orig.__dict__.items():
            if torch._C._jit_is_script_object(val) and name not in orig._parameters and (name not in orig._buffers):
                setattr(tmp_module, name, val)
        if orig._backward_hooks:
            raise ValueError("Modules that have backward hooks assigned can't be compiled: " + str(orig))
        for name, submodule in orig._modules.items():
            if submodule is None:
                continue
            tmp_module._modules[name] = make_module(submodule, TracedModule, _compilation_unit=None)
        script_module = torch.jit._recursive.create_script_module(tmp_module, lambda module: (), share_types=False, is_tracing=True)
        self.__dict__['_name'] = type(orig).__name__
        self.__dict__['_actual_script_module'] = script_module
        for name in ('_parameters', '_buffers', '_modules', 'training'):
            delattr(self, name)

    def forward(self, *args, **kwargs):
        raise RuntimeError('Trace submodules cannot be called.')

    def __getattr__(self, attr):
        if '_actual_script_module' not in self.__dict__:
            return super().__getattr__(attr)
        return getattr(self._actual_script_module, attr)

    def __setattr__(self, attr, value):
        if '_actual_script_module' not in self.__dict__:
            return super().__setattr__(attr, value)
        setattr(self._actual_script_module, attr, value)

    def _get_name(self):
        return self._name

    def extra_repr(self):
        return f'original_name={self._name}'