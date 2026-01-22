import collections
import io
import sys
import types
from typing import (
import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.distributed import _remote_device
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
def _param_rrefs(module_rref, recurse) -> List[rpc.RRef[Parameter]]:
    ret: List[rpc.RRef[Parameter]] = []
    for param in module_rref.local_value().parameters(recurse):
        ret.append(rpc.RRef(param))
    return ret