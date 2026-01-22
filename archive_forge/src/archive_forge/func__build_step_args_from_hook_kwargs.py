from collections import OrderedDict
from typing import Any, Iterator, List, Optional, Union
import torch
from lightning_utilities import WarningCache
import pytorch_lightning as pl
from lightning_fabric.utilities import move_data_to_device
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from pytorch_lightning.loops.loop import _Loop
from pytorch_lightning.loops.progress import _Progress
from pytorch_lightning.loops.utilities import _no_grad_context, _select_data_fetcher, _verify_dataloader_idx_requirement
from pytorch_lightning.overrides.distributed import _IndexBatchSamplerWrapper
from pytorch_lightning.strategies.launchers import _MultiProcessingLauncher
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.data_connector import (
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT
def _build_step_args_from_hook_kwargs(self, hook_kwargs: OrderedDict, step_hook_name: str) -> tuple:
    """Helper method to build args for `predict_step`."""
    kwargs = hook_kwargs.copy()
    step_hook_fx = getattr(self.trainer.lightning_module, step_hook_name)
    if not is_param_in_hook_signature(step_hook_fx, 'batch_idx', min_args=2):
        kwargs.pop('batch_idx', None)
    return tuple(kwargs.values())