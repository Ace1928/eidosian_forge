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
def _prepare_subcommand_kwargs(self, subcommand: str) -> Dict[str, Any]:
    """Prepares the keyword arguments to pass to the subcommand to run."""
    fn_kwargs = {k: v for k, v in self.config_init[subcommand].items() if k in self._subcommand_method_arguments[subcommand]}
    fn_kwargs['model'] = self.model
    if self.datamodule is not None:
        fn_kwargs['datamodule'] = self.datamodule
    return fn_kwargs