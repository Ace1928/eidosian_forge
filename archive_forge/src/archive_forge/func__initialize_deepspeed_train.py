import argparse
import json
import logging
import os
import platform
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Mapping, Optional, Tuple, Union
import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import ClusterEnvironment
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.deepspeed import (
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, LRScheduler, ReduceLROnPlateau
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.types import LRSchedulerConfig
def _initialize_deepspeed_train(self, model: Module) -> None:
    optimizer, scheduler = (None, None)
    assert isinstance(self.config, dict)
    if 'optimizer' in self.config:
        rank_zero_info('You have specified an optimizer and/or scheduler within the DeepSpeed config. It is recommended to define it in `LightningModule.configure_optimizers`.')
        lr_scheduler = None
    else:
        optimizer, lr_scheduler = self._init_optimizers()
        if lr_scheduler is not None:
            scheduler = lr_scheduler.scheduler
    model, deepspeed_optimizer = self._setup_model_and_optimizer(model, optimizer, scheduler)
    self._set_deepspeed_activation_checkpointing()
    self.optimizers = [deepspeed_optimizer]
    deepspeed_scheduler = model.lr_scheduler
    if deepspeed_scheduler is not None:
        model.lr_scheduler = None
        if lr_scheduler is None:
            lr_scheduler = LRSchedulerConfig(deepspeed_scheduler, interval='step')
        else:
            lr_scheduler.scheduler = deepspeed_scheduler
        self.lr_scheduler_configs = [lr_scheduler]
    self.model = model