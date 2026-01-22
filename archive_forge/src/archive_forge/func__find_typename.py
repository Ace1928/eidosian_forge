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
def _find_typename(v):
    if isinstance(v, type):
        return torch.typename(v)
    else:
        raise RuntimeError('Only type of the `nn.Module` should be passed in the set for argument `export_modules_as_functions`. Got `%s`.' % type(v).__name__)