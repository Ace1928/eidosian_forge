import os
import sys
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import _warn
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _set_seed(self) -> None:
    """Sets the seed."""
    config_seed = self._get(self.config, 'seed_everything')
    if config_seed is False:
        return
    if config_seed is True:
        config_seed = seed_everything(workers=True)
    else:
        config_seed = seed_everything(config_seed, workers=True)
    if self.subcommand:
        self.config[self.subcommand]['seed_everything'] = config_seed
    else:
        self.config['seed_everything'] = config_seed