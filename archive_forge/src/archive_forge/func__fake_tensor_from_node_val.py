from __future__ import annotations
import abc
import dataclasses
import inspect
import logging
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Set
import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.node import Node  # noqa: F401
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree
@_beartype.beartype
def _fake_tensor_from_node_val(node: torch.fx.Node) -> fake_tensor.FakeTensor:
    """Syntactic sugar for retrieving fake tensor from node.meta['val']."""
    val = node.meta.get('val', None)
    if not isinstance(val, fake_tensor.FakeTensor):
        raise RuntimeError(f'Cannot retrieve fake tensor from node {node}. Got type({type(val)}) instead.')
    return val