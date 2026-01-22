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
def STORE_ATTR(self, inst):
    speculation = self.speculate()
    if speculation.failed:
        return self.store_attr_graph_break(inst)
    val, obj = self.popn(2)
    if isinstance(obj, NNModuleVariable):
        assert not self.export, f'Mutating module attribute {inst.argval} during export.'
    try:
        BuiltinVariable(setattr).call_function(self, [obj, ConstantVariable.create(inst.argval), val], {})
        return
    except Unsupported as e:
        if not self.should_compile_partial_graph():
            raise
        log.debug('STORE_ATTR triggered compile', exc_info=True)
        e.remove_from_stats()
        e.add_to_stats('graph_break')
    speculation.fail_and_restart_analysis()