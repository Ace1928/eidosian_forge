import collections
import contextlib
import copy
import dataclasses
import dis
import functools
import importlib
import inspect
import itertools
import linecache
import logging
import operator
import sys
import textwrap
import threading
import traceback
import types
import typing
import weakref
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Type
from unittest.mock import patch
import torch
import torch._logging
from torch._guards import Checkpointable, tracing, TracingContext
from . import (
from .allowed_functions import is_allowed, is_builtin_constant, is_forbidden
from .bytecode_analysis import (
from .bytecode_transformation import (
from .code_context import code_context
from .codegen import PyCodegen
from .current_scope_id import current_scope_id
from .exc import ArgsMismatchError, BackendCompilerFailed, unimplemented, Unsupported
from .funcname_cache import get_funcname
from .guards import GuardBuilder, install_guard
from .output_graph import GraphCompileReason, OutputGraph, OutputGraphState
from .replay_record import DummyModule, ExecutionRecorder
from .resume_execution import ContinueExecutionCache, ReenterWith
from .source import (
from .utils import (
from .variables.base import (
from .variables.builder import VariableBuilder, wrap_fx_proxy
from .variables.builtin import BuiltinVariable
from .variables.constant import ConstantVariable, EnumVariable
from .variables.ctx_manager import (
from .variables.dicts import ConstDictVariable, SetVariable
from .variables.functions import (
from .variables.lists import (
from .variables.misc import (
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
from .variables.torch import TorchVariable
from .variables.user_defined import (
def IMPORT_NAME(self, inst):
    level, fromlist = self.popn(2)
    level = level.as_python_constant()
    fromlist = fromlist.as_python_constant()
    module_name = inst.argval
    recorded_name = f'{ExecutionRecorder.LOCAL_MOD_PREFIX}_{level}_{fromlist}_{module_name}'
    if recorded_name in self.f_globals:
        value = self.f_globals[recorded_name]
        source = GlobalSource(recorded_name)
    else:
        value = __import__(module_name, fromlist=fromlist, level=level, globals=self.f_globals)
        if level != 0:
            pkg = self.calc_package()
            module_name = self.resolve_name(module_name, pkg, level)
        if not fromlist:
            top_level_module_name = module_name.partition('.')[0]
            source = self.import_source(top_level_module_name)
        else:
            source = self.import_source(module_name)
    if config.replay_record_enabled:
        self.exec_recorder.add_local_mod(recorded_name, value)
    if is_allowed(value):
        self.push(TorchVariable(value, source=source))
    elif istype(value, (types.ModuleType, DummyModule)):
        self.push(PythonModuleVariable(value, source=source))
    else:
        unimplemented(f'IMPORT_NAME {typestr(value)}')