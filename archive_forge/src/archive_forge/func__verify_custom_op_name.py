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
def _verify_custom_op_name(symbolic_name: str):
    if not re.match('^[a-zA-Z0-9-_]+::[a-zA-Z-_]+[a-zA-Z0-9-_]*$', symbolic_name):
        raise errors.OnnxExporterError(f'Failed to register operator {symbolic_name}. The symbolic name must match the format domain::name, and should start with a letter and contain only alphanumerical characters')
    ns, _ = jit_utils.parse_node_kind(symbolic_name)
    if ns == 'onnx':
        raise ValueError(f'Failed to register operator {symbolic_name}. {ns} domain cannot be modified.')