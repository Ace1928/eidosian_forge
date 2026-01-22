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
class ONNXTracedModule(torch.nn.Module):

    def __init__(self, inner, strict=True, force_outplace=False, return_inputs=False, return_inputs_states=False):
        super().__init__()
        self.inner = inner
        self.strict = strict
        self._force_outplace = force_outplace
        self._return_inputs = return_inputs
        self._return_inputs_states = return_inputs_states

    def forward(self, *args: torch.Tensor):
        in_vars, in_desc = _flatten(args)
        module_state = list(_unique_state_dict(self, keep_vars=True).values())
        ret_inputs = []
        inputs_states = []
        outs = []

        def wrapper(*args):
            in_args: List[torch.Tensor] = []
            for i in range(len(in_vars)):
                if not isinstance(args[i], torch.Tensor):
                    raise RuntimeError('Expected Tensor argument')
                in_args.append(args[i])
            trace_inputs = _unflatten(in_args, in_desc)
            if self._return_inputs:
                ret_inputs.append(tuple((x.clone(memory_format=torch.preserve_format) for x in args)))
            if self._return_inputs_states:
                inputs_states.append(_unflatten(in_args, in_desc))
            outs.append(self.inner(*trace_inputs))
            if self._return_inputs_states:
                inputs_states[0] = (inputs_states[0], trace_inputs)
            out_vars, _ = _flatten(outs)
            if len(out_vars) == 1:
                return out_vars[0]
            else:
                return tuple(out_vars)
        graph, out = torch._C._create_graph_by_tracing(wrapper, in_vars + module_state, _create_interpreter_name_lookup_fn(), self.strict, self._force_outplace)
        if self._return_inputs:
            return (graph, outs[0], ret_inputs[0])
        if self._return_inputs_states:
            return (graph, outs[0], inputs_states[0])
        else:
            return (graph, outs[0])