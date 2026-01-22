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
def _make_closure(self, kwargs: OrderedDict, optimizer: Optimizer, batch_idx: int) -> Closure:
    """Build a closure object that captures the given arguments and runs the `training_step` function and
        optionally other functions such as `backward` and `zero_grad`."""
    step_fn = self._make_step_fn(kwargs)
    backward_fn = self._make_backward_fn(optimizer)
    zero_grad_fn = self._make_zero_grad_fn(batch_idx, optimizer)
    return Closure(step_fn=step_fn, backward_fn=backward_fn, zero_grad_fn=zero_grad_fn)