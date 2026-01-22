from typing import Optional, Any
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer
from .. import functional as F
from .. import init
from ._functions import SyncBatchNorm as sync_batch_norm
from .lazy import LazyModuleMixin
from .module import Module
def _check_non_zero_input_channels(self, input):
    if input.size(1) == 0:
        raise ValueError('SyncBatchNorm number of input channels should be non-zero')