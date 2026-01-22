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
def _trace_and_get_graph_from_model(model, args):
    orig_state_dict_keys = torch.jit._unique_state_dict(model).keys()
    prev_autocast_cache_enabled = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(model, args, strict=False, _force_outplace=False, _return_inputs_states=True)
    torch.set_autocast_cache_enabled(prev_autocast_cache_enabled)
    warn_on_static_input_change(inputs_states)
    if orig_state_dict_keys != torch.jit._unique_state_dict(model).keys():
        raise RuntimeError('state_dict changed after running the tracer; something weird is happening in your model!')
    return (trace_graph, torch_out)