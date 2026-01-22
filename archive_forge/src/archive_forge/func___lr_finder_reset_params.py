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
def __lr_finder_reset_params(trainer: 'pl.Trainer', num_training: int, early_stop_threshold: Optional[float]) -> None:
    from pytorch_lightning.loggers.logger import DummyLogger
    trainer.strategy.lr_scheduler_configs = []
    trainer.callbacks = [_LRCallback(num_training, early_stop_threshold, progress_bar_refresh_rate=1)]
    trainer.logger = DummyLogger() if trainer.logger is not None else None
    trainer.fit_loop.epoch_loop.max_steps = num_training + trainer.global_step
    trainer.limit_val_batches = num_training