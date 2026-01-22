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
@break_graph_if_unsupported(push=1)
def CALL_FUNCTION_EX(self, inst):
    kwargsvars: VariableTracker
    if inst.argval == 0:
        kwargsvars = ConstDictVariable({}, dict)
        argsvars = self.pop()
    elif inst.argval == 1:
        kwargsvars = self.pop()
        argsvars = self.pop()
    else:
        unimplemented('CALL_FUNCTION_EX')
    fn = self.pop()
    if sys.version_info >= (3, 11):
        null = self.pop()
        assert isinstance(null, NullVariable)
    if isinstance(fn, GetAttrVariable) and isinstance(fn.obj, TensorVariable) and (fn.name == 'view') and isinstance(argsvars, (ConstantVariable, TensorVariable)):
        argsvars = TupleVariable([argsvars])
    if not isinstance(argsvars, BaseListVariable) and argsvars.has_unpack_var_sequence(self):
        argsvars = TupleVariable(argsvars.unpack_var_sequence(self))
    if not isinstance(argsvars, BaseListVariable) or not isinstance(kwargsvars, ConstDictVariable):
        unimplemented(f'non-static call {typestr(argsvars)} {typestr(kwargsvars)}')
    self.call_function(fn, argsvars.items, kwargsvars.items)