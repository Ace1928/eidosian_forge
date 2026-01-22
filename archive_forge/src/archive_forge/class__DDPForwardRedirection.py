import logging
from contextlib import nullcontext
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union
import torch
import torch.distributed
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.optimizer import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import _IS_WINDOWS
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import ReduceOp
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.overrides.distributed import _register_ddp_comm_hook, _sync_module_states, prepare_for_backward
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies.launchers import _MultiProcessingLauncher, _SubprocessScriptLauncher
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import TBroadcast, _ForwardRedirection
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import _augment_message
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_info, rank_zero_only
class _DDPForwardRedirection(_ForwardRedirection):

    @override
    def on_after_inner_forward(self, wrapper_module: Module, original_module: 'pl.LightningModule') -> None:
        if isinstance(wrapper_module, DistributedDataParallel) and (not original_module.automatic_optimization):
            wrapper_module.require_backward_grad_sync = False

    @override
    def on_after_outer_forward(self, wrapper_module: Module, original_module: 'pl.LightningModule') -> None:
        if isinstance(wrapper_module, DistributedDataParallel) and (not original_module.automatic_optimization):
            wrapper_module.require_backward_grad_sync = True