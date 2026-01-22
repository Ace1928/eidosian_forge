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
def _recursive_script_module_receiver(recursive_script_module_serialized):
    """Deserializes a RecursiveScriptModule that does not contain a script RemoteModule."""
    f = io.BytesIO(recursive_script_module_serialized)
    m = torch.jit.load(f)
    return m