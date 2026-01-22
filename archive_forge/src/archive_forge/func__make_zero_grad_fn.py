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
def _make_zero_grad_fn(self, batch_idx: int, optimizer: Optimizer) -> Optional[Callable[[], None]]:
    """Build a `zero_grad` function that zeroes the gradients before back-propagation.

        Returns ``None`` in the case backward needs to be skipped.

        """
    if self._skip_backward:
        return None
    is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0
    if not is_first_batch_to_accumulate:
        return None

    def zero_grad_fn() -> None:
        self._on_before_zero_grad(optimizer)
        self._optimizer_zero_grad(batch_idx, optimizer)
    return zero_grad_fn