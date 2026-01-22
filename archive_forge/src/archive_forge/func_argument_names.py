from __future__ import annotations
import contextlib
import dis
import functools
import inspect
import logging
import os
import sys
import textwrap
import threading
import traceback
import types
import warnings
from dataclasses import dataclass
from enum import Enum
from os.path import dirname, join
from typing import (
from unittest.mock import patch
import torch
import torch.fx
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch import _guards
from torch._subclasses import fake_tensor
from torch.export import Constraint
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.nn.parallel.distributed import DistributedDataParallel
from ..fx import GraphModule
from .backends.registry import CompilerFn, lookup_backend
from .hooks import Hooks
from . import config, convert_frame, external_utils, skipfiles, utils
from .code_context import code_context
from .exc import CondOpArgsMismatchError, UserError, UserErrorType
from .mutation_guard import install_generation_tagging_init
from .types import CacheEntry, DynamoCallback
from .utils import compile_times
from torch._dispatch.python import enable_python_dispatcher
from torch.utils._python_dispatch import _disable_current_modes
import sympy
def argument_names(f_sig, args, kwargs) -> List[str]:

    def signature_to_fullargspec(sig: inspect.Signature):
        params = list(sig.parameters.values())
        args = [p.name for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        kwonlyargs = [p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY]
        varargs = next((p.name for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL), None)
        varkw = next((p.name for p in params if p.kind == inspect.Parameter.VAR_KEYWORD), None)
        defaults = tuple((p.default for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is not inspect.Parameter.empty))
        kwonlydefaults = {p.name: p.default for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty}
        annotations = {}
        if sig.return_annotation:
            annotations = {'return': sig.return_annotation}
        for parameter in params:
            annotations[parameter.name] = parameter.annotation
        return inspect.FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)
    fullargspec = signature_to_fullargspec(f_sig)
    input_strs = fullargspec.args[:len(args)]
    if len(args) > len(fullargspec.args):
        assert fullargspec.varargs is not None, 'More arguments than expected'
        input_strs += [f'{fullargspec.varargs}_{i}' for i in range(0, len(args) - len(input_strs))]
    elif len(args) < len(fullargspec.args):
        for unprovided_arg in fullargspec.args[len(args):-len(fullargspec.defaults or [])]:
            assert unprovided_arg in kwargs, f'Missing argument {unprovided_arg}'
    input_strs += list(kwargs.keys())
    for kwonly_arg in fullargspec.kwonlyargs:
        kwonlydefaults = fullargspec.kwonlydefaults or {}
        assert kwonly_arg in kwargs or kwonly_arg in kwonlydefaults, f'Missing keyword only argument {kwonly_arg}'
    return input_strs