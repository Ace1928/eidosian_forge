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
def _select_data_fetcher(trainer: 'pl.Trainer', stage: RunningStage) -> _DataFetcher:
    lightning_module = trainer.lightning_module
    if stage == RunningStage.TESTING:
        step_fx_name = 'test_step'
    elif stage == RunningStage.TRAINING:
        step_fx_name = 'training_step'
    elif stage in (RunningStage.VALIDATING, RunningStage.SANITY_CHECKING):
        step_fx_name = 'validation_step'
    elif stage == RunningStage.PREDICTING:
        step_fx_name = 'predict_step'
    else:
        raise RuntimeError(f'DataFetcher is unsupported for {trainer.state.stage}')
    step_fx = getattr(lightning_module, step_fx_name)
    if is_param_in_hook_signature(step_fx, 'dataloader_iter', explicit=True):
        rank_zero_warn(f'Found `dataloader_iter` argument in the `{step_fx_name}`. Note that the support for this signature is experimental and the behavior is subject to change.')
        return _DataLoaderIterDataFetcher()
    return _PrefetchDataFetcher()