from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
@_beartype.beartype
def _unpack_list(list_value: _C.Value) -> List[_C.Value]:
    list_node = list_value.node()
    if list_node.kind() != 'prim::ListConstruct':
        raise errors.SymbolicValueError(f"ONNX symbolic expected node type prim::ListConstruct, got '{list_node}'.", list_value)
    return list(list_node.inputs())