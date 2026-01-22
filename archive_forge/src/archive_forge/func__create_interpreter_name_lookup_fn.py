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
def _create_interpreter_name_lookup_fn(frames_up=1):

    def _get_interpreter_name_for_var(var):
        frame = inspect.currentframe()
        if not frame:
            raise RuntimeError('failed to inspect frame')
        i = 0
        while i < frames_up + 1:
            frame = frame.f_back
            if not frame:
                raise RuntimeError('failed to get frame')
            i += 1
        f_locals = frame.f_locals
        f_globals = frame.f_globals
        for k, v in f_locals.items():
            if isinstance(v, torch.Tensor) and var is v:
                return k if k != 'self' else ''
        return ''
    return _get_interpreter_name_for_var