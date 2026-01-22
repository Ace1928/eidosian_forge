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
def _convert_torch_args_to_onnxfunction_args(param_schemas: Sequence[onnxscript.values.ParamSchema], args: List[fx_type_utils.Argument], kwargs: Dict[str, fx_type_utils.Argument], allow_extra_kwargs: bool=False) -> Tuple[List[Any], Dict[str, Any]]:
    """Convert Python args and kwargs to OnnxFunction acceptable with matching ONNX ParamSchema.

    NOTE: This is different from the param_schema separating in dispatcher, since at this point
    we are already sure that the args and kwargs are in order and matched.

    Args:
        param_schemas: The parameter schemas of an Op or a OnnxFunction.
        args: The Python positional arguments supplied by the caller.
        kwargs: The Python keyword arguments supplied by the caller.
        allow_extra_kwargs: Whether to allow extra keyword arguments.
            When set to True, extra/unknown arguments will be ignored.

    Returns:
        A tuple of two elements:
        - A list of Python positional argument.
        - An ordered dictionary of Python keyword argument names and its values.

    Raises:
        TypeError: When allow_extra_kwargs is False and there are unknown kwargs.
        TypeError: When a required input is not provided.
    """
    all_param_names = {param.name for param in param_schemas}
    extra_kwargs = set(kwargs).difference(all_param_names)
    if extra_kwargs and (not allow_extra_kwargs):
        raise TypeError(f"Unexpected keyword arguments '{extra_kwargs}'")
    tagged_args: list[Any] = []
    tagged_kwargs: dict[str, Any] = {}
    for i, param in enumerate(param_schemas):
        if param.is_variadic_input:
            tagged_args.extend((arg for arg in args[i:]))
            args = []
            continue
        if i < len(args):
            if param.is_input or isinstance(args[i], torch.dtype):
                tagged_args.append(_convert_tensor_to_numpy(args[i]))
            else:
                tagged_args.append(args[i])
        elif param.name in kwargs:
            if param.is_input:
                tagged_kwargs[param.name] = _convert_tensor_to_numpy(kwargs[param.name])
            else:
                tagged_kwargs[param.name] = kwargs[param.name]
        elif param.required:
            raise TypeError(f"Required input/attribute '{param}' was not provided")
    return (tagged_args, tagged_kwargs)