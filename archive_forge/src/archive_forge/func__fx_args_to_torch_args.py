from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]
import torch
import torch.fx
from torch.fx.experimental import symbolic_shapes
from torch.onnx import _constants, _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
def _fx_args_to_torch_args(fx_args: List[fx_type_utils.Argument], fx_graph_module: torch.fx.GraphModule) -> List[fx_type_utils.Argument]:
    """Recursively convert fx args to torch args"""
    wrapped_args: List[fx_type_utils.Argument] = []
    for arg in fx_args:
        if isinstance(arg, torch.fx.Node):
            fake_tensor = arg.meta.get('val')
            if fake_tensor is None and arg.op == 'get_attr':
                fake_tensor = getattr(fx_graph_module, arg.target)
            if isinstance(fake_tensor, torch.Tensor):
                real_tensor = generate_random_tensors(fake_tensor.shape, fake_tensor.dtype)
                wrapped_args.append(real_tensor)
            elif isinstance(fake_tensor, (int, float, bool)):
                wrapped_args.append(fake_tensor)
            elif symbolic_shapes.has_hint(fake_tensor):
                wrapped_args.append(symbolic_shapes.hint_int(fake_tensor))
            else:
                raise ValueError(f"Unexpected input argument type found inside fx.Node. arg: {arg}; arg.meta['val']/get_attr: {fake_tensor}; type(arg.meta['val']/get_attr): {type(fake_tensor)}.")
        elif isinstance(arg, Sequence):
            wrapped_args.append(_fx_args_to_torch_args(arg, fx_graph_module))
        elif isinstance(arg, (int, float, torch.dtype)) or arg is None:
            wrapped_args.append(arg)
        elif isinstance(arg, torch.device):
            wrapped_args.append(str(arg))
        else:
            raise ValueError(f'Unexpected input argument type is found in node arguments. arg: {arg}; ')
    return wrapped_args