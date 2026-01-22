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
def _get_batch_indices(self, dataloader: object) -> List[List[int]]:
    """Returns a reference to the seen batch indices if the dataloader has a batch sampler wrapped by our
        :class:`~pytorch_lightning.overrides.distributed._IndexBatchSamplerWrapper`."""
    batch_sampler = getattr(dataloader, 'batch_sampler', None)
    if not isinstance(batch_sampler, _IndexBatchSamplerWrapper):
        self._warning_cache.warn(f"Couldn't infer the batch indices fetched from your dataloader: `{type(dataloader).__name__}`")
        return []
    return batch_sampler.seen_batch_indices