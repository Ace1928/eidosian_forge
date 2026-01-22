from __future__ import annotations
import dataclasses
import re
import typing
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, registration
@_beartype.beartype
def _add_attribute(node: _C.Node, key: str, value: Any, aten: bool):
    """Initializes the right attribute based on type of value."""
    m = _ATTR_PATTERN.match(key)
    if m is None:
        raise ValueError(f"Invalid attribute specifier '{key}' names must be suffixed with type, e.g. 'dim_i' or 'dims_i'")
    name, kind = (m.group(1), m.group(2))
    if _is_onnx_list(value):
        kind += 's'
    if aten and _is_caffe2_aten_fallback():
        if isinstance(value, torch.Tensor):
            if value.numel() > 1:
                raise ValueError('Should not pass tensor attribute')
            value = _scalar(value)
            if isinstance(value, float):
                kind = 'f'
            else:
                kind = 'i'
    return getattr(node, f'{kind}_')(name, value)