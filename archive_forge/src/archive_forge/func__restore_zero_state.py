import argparse
import json
import logging
import os
import platform
from contextlib import ExitStack
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Mapping, Optional, Tuple, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
from lightning_fabric.accelerators import Accelerator, CUDAAccelerator
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.ddp import DDPStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import _Sharded
from lightning_fabric.utilities.distributed import log
from lightning_fabric.utilities.load import _move_state_into
from lightning_fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH
def _restore_zero_state(self, module: Module, ckpt: Mapping[str, Any]) -> None:
    """Overrides the normal load_state_dict behaviour in PyTorch to ensure we gather parameters that may be sharded
        across processes before loading the state dictionary when using ZeRO stage 3. This is then automatically synced
        across processes.

        Args:
            ckpt: The ckpt file.

        """
    import deepspeed

    def load(module: torch.nn.Module, prefix: str='') -> None:
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []
        state_dict = ckpt['state_dict']
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
            if self.is_global_zero:
                module._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=True, missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(module, prefix='')