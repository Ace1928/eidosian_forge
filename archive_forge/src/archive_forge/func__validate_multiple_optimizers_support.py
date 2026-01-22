from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, overload
from weakref import proxy
import torch
from torch import optim
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import Optimizable, ReduceLROnPlateau, _Stateful
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import LRSchedulerConfig, LRSchedulerTypeTuple
def _validate_multiple_optimizers_support(optimizers: List[Optimizer], model: 'pl.LightningModule') -> None:
    if is_param_in_hook_signature(model.training_step, 'optimizer_idx', explicit=True):
        raise RuntimeError('Training with multiple optimizers is only supported with manual optimization. Remove the `optimizer_idx` argument from `training_step`, set `self.automatic_optimization = False` and access your optimizers in `training_step` with `opt1, opt2, ... = self.optimizers()`.')
    if model.automatic_optimization and len(optimizers) > 1:
        raise RuntimeError('Training with multiple optimizers is only supported with manual optimization. Set `self.automatic_optimization = False`, then access your optimizers in `training_step` with `opt1, opt2, ... = self.optimizers()`.')