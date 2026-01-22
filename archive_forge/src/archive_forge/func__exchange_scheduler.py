import importlib
import logging
import os
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRScheduler, LRSchedulerConfig
def _exchange_scheduler(self, trainer: 'pl.Trainer') -> None:
    """Decorate `trainer.strategy.setup_optimizers` method such that it sets the user's originally specified
        optimizer together with a new scheduler that takes care of the learning rate search."""
    from pytorch_lightning.core.optimizer import _validate_optimizers_attached
    optimizers = trainer.strategy.optimizers
    if len(optimizers) != 1:
        raise MisconfigurationException(f'`model.configure_optimizers()` returned {len(optimizers)}, but learning rate finder only works with single optimizer')
    optimizer = optimizers[0]
    new_lrs = [self.lr_min] * len(optimizer.param_groups)
    for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
        param_group['lr'] = new_lr
        param_group['initial_lr'] = new_lr
    args = (optimizer, self.lr_max, self.num_training)
    scheduler = _LinearLR(*args) if self.mode == 'linear' else _ExponentialLR(*args)
    scheduler = cast(LRScheduler, scheduler)
    trainer.strategy.optimizers = [optimizer]
    trainer.strategy.lr_scheduler_configs = [LRSchedulerConfig(scheduler, interval='step')]
    _validate_optimizers_attached(trainer.optimizers, trainer.lr_scheduler_configs)