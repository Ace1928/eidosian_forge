from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@_beartype.beartype
def _setup_trace_module_map(model: Union[torch.nn.Module, torch.jit.ScriptModule], export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]]]) -> Set[str]:

    def __register_attribute_hook():
        attr_name = '_onnx_attrs'

        def _track_module_attributes_forward_pre_hook(module, input):
            setattr(module, attr_name, _get_module_attributes(module))

        def _track_module_attributes_forward_hook(module, input, output):
            tracing_state = _C._get_tracing_state()
            if not tracing_state:
                return
            graph = tracing_state.graph()
            onnx_attrs = {}
            if hasattr(module, attr_name):
                onnx_attrs = getattr(module, attr_name)
                delattr(module, attr_name)
            _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)
        for m in model.modules():
            m.register_forward_hook(_track_module_attributes_forward_hook)
            m.register_forward_pre_hook(_track_module_attributes_forward_pre_hook)

    def _unqualified_variable_name(qualified_name: str) -> str:
        """
        Parse qualified variable name and return the unqualified version.

        Pure numeric atoms are considered inadequate, so this function will look past them,
        and start from the first non-numeric atom.

        Example:
            >>> _unqualified_variable_name('__main__.Foo.bar')
            'bar'
            >>> _unqualified_variable_name('__main__.Foo.bar.0')
            'bar.0'
        """
        name_atoms = qualified_name.split('.')
        for i, atom in reversed(list(enumerate(name_atoms))):
            if not atom.isnumeric():
                return '.'.join(name_atoms[i:])
        return qualified_name
    trace_module_map = {_m: torch._C._jit_onnx_create_full_scope_name(torch.typename(type(_m)), _unqualified_variable_name(_n)) for _n, _m in model.named_modules()}
    torch.jit._trace._trace_module_map = trace_module_map
    if isinstance(export_modules_as_functions, bool) and export_modules_as_functions:
        module_typenames = {torch.typename(type(module)) for module in trace_module_map}
    elif isinstance(export_modules_as_functions, set) and export_modules_as_functions:

        def _find_typename(v):
            if isinstance(v, type):
                return torch.typename(v)
            else:
                raise RuntimeError('Only type of the `nn.Module` should be passed in the set for argument `export_modules_as_functions`. Got `%s`.' % type(v).__name__)
        module_typenames = {_find_typename(v) for v in export_modules_as_functions}
    else:
        module_typenames = set()
    if module_typenames:
        __register_attribute_hook()
    return module_typenames