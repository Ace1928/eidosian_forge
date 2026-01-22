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
def _training_step(self, kwargs: OrderedDict) -> ClosureResult:
    """Performs the actual train step with the tied hooks.

        Args:
            kwargs: the kwargs passed down to the hooks.

        Returns:
            A ``ClosureResult`` containing the training step output.

        """
    trainer = self.trainer
    training_step_output = call._call_strategy_hook(trainer, 'training_step', *kwargs.values())
    self.trainer.strategy.post_training_step()
    return self.output_result_cls.from_training_step_output(training_step_output, trainer.accumulate_grad_batches)