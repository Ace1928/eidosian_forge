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
def _format_batch_size_and_grad_accum_config(self) -> None:
    assert isinstance(self.config, dict)
    if self.lightning_module is None:
        return
    if 'gradient_accumulation_steps' in self.config:
        raise MisconfigurationException('Do not set `gradient_accumulation_steps` in the DeepSpeed config as this will be set with the `accumulate_grad_batches` argument passed via the Lightning Trainer.')
    self.config['gradient_accumulation_steps'] = self.lightning_module.trainer.accumulate_grad_batches
    if 'train_micro_batch_size_per_gpu' not in self.config:
        batch_size = self._auto_select_batch_size()
        self.config['train_micro_batch_size_per_gpu'] = batch_size
    if 'gradient_clipping' not in self.config:
        self.config['gradient_clipping'] = self.lightning_module.trainer.gradient_clip_val or 0.0