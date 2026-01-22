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
def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
    """Adds arguments from the core classes to the parser."""
    parser.add_lightning_class_args(self.trainer_class, 'trainer')
    trainer_defaults = {'trainer.' + k: v for k, v in self.trainer_defaults.items() if k != 'callbacks'}
    parser.set_defaults(trainer_defaults)
    parser.add_lightning_class_args(self._model_class, 'model', subclass_mode=self.subclass_mode_model)
    if self.datamodule_class is not None:
        parser.add_lightning_class_args(self._datamodule_class, 'data', subclass_mode=self.subclass_mode_data)
    else:
        parser.add_lightning_class_args(self._datamodule_class, 'data', subclass_mode=self.subclass_mode_data, required=False)