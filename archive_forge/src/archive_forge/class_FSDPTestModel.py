import itertools
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest import mock
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.fully_sharded_data_parallel import (
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, wrap
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import FILE_SCHEMA, get_cycles_per_ms
class FSDPTestModel(nn.Module, ABC):
    """This defines the interface expected from all models used commonly for
    FSDP unit tests."""

    @abstractmethod
    def get_input(self, device) -> Tuple[torch.Tensor, ...]:
        """Returns an input for the model as as tuple."""
        ...

    @abstractmethod
    def get_loss(self, input, output) -> torch.Tensor:
        """Returns the loss given the input and output."""
        ...

    @abstractmethod
    def run_backward(self, loss) -> None:
        """Runs the backward pass (e.g. including ``loss.backward()``)."""
        ...

    @staticmethod
    @abstractmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, *init_args: Any, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False, **init_kwargs: Any) -> nn.Module:
        """Initializes an instance of this model."""
        ...