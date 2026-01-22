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
class NestedWrappedModule(FSDPTestModel):

    def __init__(self, group: dist.ProcessGroup, wrap_fsdp: bool, cuda_init_mode: CUDAInitMode, deterministic: bool, **fsdp_kwargs):
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE

        def _maybe_wrap(layer):
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer
        if deterministic:
            torch.manual_seed(0)
        self.module = nn.Sequential(_maybe_cuda(nn.Linear(8, 4), move_to_cuda), _maybe_wrap(nn.Sequential(_maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)), _maybe_cuda(nn.Linear(16, 16), move_to_cuda))), _maybe_wrap(_maybe_cuda(nn.Linear(16, 4), move_to_cuda)), _maybe_cuda(nn.Linear(4, 8), move_to_cuda))

    def get_input(self, device):
        torch.manual_seed(1 + self.rank)
        return (torch.rand(4, 8, device=device),)

    def forward(self, x):
        return self.module(x)

    def get_loss(self, input, output):
        loss = output.sum()
        return loss

    def run_backward(self, loss):
        loss.backward()

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False) -> nn.Module:
        """
        Initializes a :class:`NestedWrappedModule` instance.

        Args:
            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap
                any modules with FSDP. If ``RECURSIVE``, then wraps some nested
                modules with FSDP but not the top-level module. The model may
                later be wrapped with a top-level FSDP external to this method
                if desired.
            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.
            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments
                forwarded to the FSDP constructor.
            deterministic (bool): Whether to make the model deterministic
                across constructions.
        """
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return NestedWrappedModule(group, wrap_fsdp=False, cuda_init_mode=cuda_init_mode, deterministic=deterministic)
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            fsdp_model = NestedWrappedModule(group, wrap_fsdp=True, cuda_init_mode=cuda_init_mode, deterministic=deterministic, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        raise ValueError(f'Unsupported FSDP init mode: {fsdp_init_mode}')