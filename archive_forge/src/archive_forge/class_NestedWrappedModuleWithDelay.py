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
class NestedWrappedModuleWithDelay(ModuleWithDelay):

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode=CUDAInitMode.CUDA_AFTER, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False, delay_after_loss_ms: int=0, delay_before_reduction_ms: int=0):
        return super(NestedWrappedModuleWithDelay, NestedWrappedModuleWithDelay).init(NestedWrappedModule, group=group, fsdp_init_mode=fsdp_init_mode, cuda_init_mode=cuda_init_mode, fsdp_kwargs=fsdp_kwargs, deterministic=deterministic, delay_after_loss_ms=delay_after_loss_ms, delay_before_reduction_ms=delay_before_reduction_ms)