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
def unpack_quantized_tensor(value, cast_onnx_accepted=True):
    if isinstance(value, torch.Tensor) and value.dtype in _qtype_vtype_map:
        q_value_dequantize = value.dequantize()
        q_scale = torch.tensor(value.q_scale(), dtype=torch.double) if cast_onnx_accepted else torch.tensor(value.q_scale(), dtype=torch.float32)
        q_zero_point = torch.tensor(value.q_zero_point(), dtype=torch.int64) if cast_onnx_accepted else torch.tensor(value.q_zero_point(), dtype=_qtype_vtype_map[value.dtype])
        q_value = q_value_dequantize / q_scale + q_zero_point
        q_value = q_value.to(dtype=_qtype_vtype_map[value.dtype])
        return (q_value, q_scale, q_zero_point)
    else:
        return (value,)