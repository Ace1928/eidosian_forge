import logging
import shutil
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Mapping, Optional, Set, Type, Union
import torch
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.fsdp import (
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.load import _lazy_load, _materialize_tensors
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, ReduceOp
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.plugins.precision.fsdp import FSDPPrecision
from pytorch_lightning.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn
@override
def _setup_model(self, model: Module) -> Module:
    """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module."""
    from torch.distributed.fsdp import FullyShardedDataParallel
    if any((isinstance(mod, FullyShardedDataParallel) for mod in model.modules())):
        if _has_meta_device_parameters(model):
            rank_zero_warn('The model is already wrapped in `FSDP` but there are still parameters on the meta device.')
        if 'auto_wrap_policy' in self.kwargs:
            rank_zero_warn('A FSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored.')
            del self.kwargs['auto_wrap_policy']
    else:
        log.debug(f'setting up FSDP model with device id: {self.root_device.index}, kwargs: {self.kwargs}')
        model = FullyShardedDataParallel(module=model, cpu_offload=self.cpu_offload, mixed_precision=self.mixed_precision_config, sharding_strategy=self.sharding_strategy, device_id=self.root_device.index, **self.kwargs)
    _move_torchmetrics_to_device(model, self.root_device)
    _setup_activation_checkpointing(model, self._activation_checkpointing_kwargs)
    return model