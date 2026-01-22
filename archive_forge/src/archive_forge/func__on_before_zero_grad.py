from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, OrderedDict
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.loops.loop import _Loop
from pytorch_lightning.loops.optimization.closure import AbstractClosure, OutputResult
from pytorch_lightning.loops.progress import _OptimizationProgress
from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior
from pytorch_lightning.trainer import call
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
    """Calls the ``on_before_zero_grad`` hook.

        Args:
            optimizer: the current optimizer

        """
    trainer = self.trainer
    self.optim_progress.optimizer.zero_grad.increment_ready()
    call._call_callback_hooks(trainer, 'on_before_zero_grad', optimizer)
    call._call_lightning_module_hook(trainer, 'on_before_zero_grad', optimizer)
    self.optim_progress.optimizer.zero_grad.increment_started()