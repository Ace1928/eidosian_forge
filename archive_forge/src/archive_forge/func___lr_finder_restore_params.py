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
def __lr_finder_restore_params(trainer: 'pl.Trainer', params: Dict[str, Any]) -> None:
    trainer.strategy.optimizers = params['optimizers']
    trainer.strategy.lr_scheduler_configs = params['lr_scheduler_configs']
    trainer.callbacks = params['callbacks']
    trainer.loggers = params['loggers']
    loop = trainer.fit_loop
    loop.epoch_loop.max_steps = params['max_steps']
    trainer.limit_val_batches = params['limit_val_batches']
    loop.load_state_dict(deepcopy(params['loop_state_dict']))
    loop.restarting = False
    trainer.should_stop = False