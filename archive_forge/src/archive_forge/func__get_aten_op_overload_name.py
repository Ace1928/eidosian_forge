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
def _get_aten_op_overload_name(n: _C.Node) -> str:
    schema = n.schema()
    if not schema.startswith('aten::') or symbolic_helper.is_caffe2_aten_fallback():
        return ''
    return _C.parse_schema(schema).overload_name