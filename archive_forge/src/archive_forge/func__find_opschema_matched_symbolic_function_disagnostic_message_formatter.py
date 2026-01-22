from __future__ import annotations
import logging
import operator
import types
from typing import (
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
@_beartype.beartype
def _find_opschema_matched_symbolic_function_disagnostic_message_formatter(fn: Callable, self, node: torch.fx.Node, default_and_custom_functions: List[registration.ONNXFunction], *args, **kwargs) -> str:
    """Format the diagnostic message for the nearest match warning."""
    all_function_overload_names = ''
    for symbolic_func in default_and_custom_functions:
        overload_func = symbolic_func.onnx_function
        all_function_overload_names += f'ONNX Node: {overload_func.name}[opset={overload_func.opset};is_custom={symbolic_func.is_custom}]. \n'
    return f'FX Node: {node.target}. \n{all_function_overload_names}'