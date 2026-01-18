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
Convert Python args and kwargs to OnnxFunction acceptable with matching ONNX ParamSchema.

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
    