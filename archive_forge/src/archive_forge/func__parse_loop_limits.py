import inspect
from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Generator, Optional, Tuple, Type
import torch
import torch.distributed as dist
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.utilities.distributed import _distributed_is_initialized
from lightning_fabric.utilities.imports import _TORCH_EQUAL_2_0
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.accelerators.xla import XLAAccelerator
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loops import _Loop
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher, _PrefetchDataFetcher
from pytorch_lightning.loops.progress import _BaseProgress
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def _parse_loop_limits(min_steps: Optional[int], max_steps: int, min_epochs: Optional[int], max_epochs: Optional[int], trainer: 'pl.Trainer') -> Tuple[int, int]:
    """This utility computes the default values for the minimum and maximum number of steps and epochs given the values
    the user has selected.

    Args:
        min_steps: Minimum number of steps.
        max_steps: Maximum number of steps.
        min_epochs: Minimum number of epochs.
        max_epochs: Maximum number of epochs.
        trainer: Trainer instance.

    Returns:
        The parsed limits, with default values being set for the ones that the user did not specify.

    """
    if max_epochs is None:
        if max_steps == -1 and (not any((isinstance(cb, Timer) for cb in trainer.callbacks))):
            rank_zero_warn('`max_epochs` was not set. Setting it to 1000 epochs. To train without an epoch limit, set `max_epochs=-1`.', category=PossibleUserWarning)
            max_epochs = 1000
        else:
            max_epochs = -1
    if min_epochs is None and min_steps is not None:
        min_epochs = 1
    if min_epochs is None:
        min_epochs = 0
    return (min_epochs, max_epochs)