from contextlib import nullcontext
from datetime import timedelta
from typing import Any, ContextManager, Dict, List, Literal, Optional, Union
import torch
import torch.distributed
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning_fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning_fabric.strategies.parallel import ParallelStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import TBroadcast, _BackwardSyncControl
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.rank_zero import rank_zero_only
class _DDPBackwardSyncControl(_BackwardSyncControl):

    @override
    def no_backward_sync(self, module: Module) -> ContextManager:
        """Blocks gradient synchronization inside the :class:`~torch.nn.parallel.distributed.DistributedDataParallel`
        wrapper."""
        if not isinstance(module, DistributedDataParallel):
            raise TypeError(f'Blocking backward sync is only possible if the module passed to `{self.__class__.__name__}.no_backward_sync` is wrapped in `DistributedDataParallel`. Got: {module.__class__.__name__}.')
        return module.no_sync()