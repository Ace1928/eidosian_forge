import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
def _infer_ep_from_device(*args) -> Tuple[str, ...]:
    """Return the first valid device (i.e., GPU or CPU) in argument list."""
    eps = []
    for arg in args:
        if hasattr(arg, 'device'):
            device = arg.device
            if device.type == 'cuda':
                eps.append('CUDAExecutionProvider')
            elif device.type == 'cpu':
                eps.append('CPUExecutionProvider')
    return tuple(eps)