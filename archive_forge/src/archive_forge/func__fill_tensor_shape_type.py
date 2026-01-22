from __future__ import annotations
import inspect
import logging
import operator
import re
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
import torch
import torch.fx
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
@_beartype.beartype
def _fill_tensor_shape_type(onnxscript_values: Union[onnxscript_graph_building.TorchScriptTensor, Tuple[onnxscript_graph_building.TorchScriptTensor, ...]], name: str, expected_values: Union[fx_type_utils.META_VALUE_TYPE, List[fx_type_utils.META_VALUE_TYPE], Tuple[Optional[fx_type_utils.META_VALUE_TYPE], ...]]):
    """Fill the meta information of onnxscript_values with that from the fx FakeTensor."""
    if isinstance(expected_values, (list, tuple)) and (not isinstance(onnxscript_values, (list, tuple))):
        return
    flat_onnxscript_values, _ = _pytree.tree_flatten(onnxscript_values)
    flat_expected_values, _ = _pytree.tree_flatten(expected_values)
    for i, (onnxscript_value, expected_value) in enumerate(zip(flat_onnxscript_values, flat_expected_values)):
        if expected_value is None:
            continue
        elif fx_type_utils.is_torch_symbolic_type(expected_value):
            onnxscript_value.dtype = fx_type_utils.from_sym_value_to_torch_dtype(expected_value)
            onnxscript_value.shape = torch.Size([1])
        elif isinstance(expected_value, (int, float, bool)):
            onnxscript_value.dtype = fx_type_utils.from_scalar_type_to_torch_dtype(type(expected_value))
            onnxscript_value.shape = torch.Size([])
        elif fx_type_utils.is_torch_complex_dtype(expected_value.dtype):
            onnxscript_value.shape = torch.Size((*expected_value.size(), 2))
            onnxscript_value.dtype = fx_type_utils.from_complex_to_float(expected_value.dtype)
            onnxscript_value.is_complex = True
        else:
            onnxscript_value.shape = expected_value.size()
            onnxscript_value.dtype = expected_value.dtype
        if i > 0:
            onnxscript_value.name = f'{name}_{i}'
        else:
            onnxscript_value.name = name