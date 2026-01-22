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
def STORE_DEREF(self, inst):
    if inst.argval in self.closure_cells:
        cell = self.closure_cells[inst.argval]
        val = self.pop()
        if isinstance(cell, ClosureVariable):
            if not self.output.is_root_tracer():
                unimplemented('HigherOrderOperator: Mutating a variable not in the current scope (ClosureVariable)')
            self.output.root_tx.symbolic_locals[cell.name] = val
        else:
            self.output.side_effects.store_cell(cell, val)
    else:
        maybe_cell = self.symbolic_locals.get(inst.argval)
        if isinstance(maybe_cell, variables.NewCellVariable):
            self.output.side_effects.store_cell(self.symbolic_locals[inst.argval], self.pop())
        else:
            if maybe_cell is not None and maybe_cell.source.name() not in self.output.root_tx.mutated_closure_cell_contents:
                self.output.root_tx.mutated_closure_cell_contents.add(maybe_cell.source.name())
                raise exc.UnspecializeRestartAnalysis()
            unimplemented('write to __closure__ while inlining')