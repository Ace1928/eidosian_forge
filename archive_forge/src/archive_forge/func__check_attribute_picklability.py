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
def _check_attribute_picklability(self):
    """Check if all the attribute has explicitly defined whether to be pickled (i.e., picklability)."""
    for k in self.__dict__.keys():
        if k not in _REMOTE_MODULE_PICKLED_ATTRIBUTES and k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING:
            raise AttributeError(f'Attribute {k} must be either in ``_REMOTE_MODULE_PICKLED_ATTRIBUTES`` or ``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``.')