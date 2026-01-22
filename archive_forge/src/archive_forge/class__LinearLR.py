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
class _LinearLR(_TORCH_LRSCHEDULER):
    """Linearly increases the learning rate between two boundaries over a number of iterations.

    Args:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer: torch.optim.Optimizer, end_lr: float, num_iter: int, last_epoch: int=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> List[float]:
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        if self.last_epoch > 0:
            val = [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]
        else:
            val = list(self.base_lrs)
        self._lr = val
        return val

    @property
    def lr(self) -> Union[float, List[float]]:
        return self._lr