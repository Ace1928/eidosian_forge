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
def _optimizer_step(self, batch_idx: int, train_step_and_backward_closure: Callable[[], Optional[Tensor]]) -> None:
    """Performs the optimizer step and some sanity checking.

        Args:
            batch_idx: the index of the current batch
            train_step_and_backward_closure: the closure function performing the train step and computing the
                gradients. By default, called by the optimizer (if possible)

        """
    trainer = self.trainer
    optimizer = trainer.strategy._lightning_optimizers[0]
    should_accumulate = trainer.fit_loop._should_accumulate()
    if not should_accumulate:
        self.optim_progress.optimizer.step.increment_ready()
    call._call_lightning_module_hook(trainer, 'optimizer_step', trainer.current_epoch, batch_idx, optimizer, train_step_and_backward_closure)
    if not should_accumulate:
        self.optim_progress.optimizer.step.increment_completed()