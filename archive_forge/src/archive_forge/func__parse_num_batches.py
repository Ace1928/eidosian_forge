import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple, Union
import torch.multiprocessing as mp
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from lightning_fabric.utilities.data import (
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from pytorch_lightning.overrides.distributed import UnrepeatedDistributedSamplerWrapper
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.data import _is_dataloader_shuffled, _update_dataloader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _graphcore_available_and_importable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities.warnings import PossibleUserWarning
def _parse_num_batches(stage: RunningStage, length: Union[int, float], limit_batches: Union[int, float]) -> Union[int, float]:
    if length == 0:
        return int(length)
    num_batches = length
    if isinstance(limit_batches, int):
        num_batches = min(length, limit_batches)
    elif isinstance(limit_batches, float) and length != float('inf'):
        num_batches = int(length * limit_batches)
    elif limit_batches != 1.0:
        raise MisconfigurationException(f'When using an `IterableDataset`, `Trainer(limit_{stage.dataloader_prefix}_batches)` must be `1.0` or an int. An int specifies `num_{stage.dataloader_prefix}_batches` to use.')
    if num_batches == 0 and limit_batches > 0.0 and isinstance(limit_batches, float) and (length != float('inf')):
        min_percentage = 1.0 / length
        raise MisconfigurationException(f'You requested to check {limit_batches} of the `{stage.dataloader_prefix}_dataloader` but {limit_batches} * {length} < 1. Please increase the `limit_{stage.dataloader_prefix}_batches` argument. Try at least `limit_{stage.dataloader_prefix}_batches={min_percentage}`')
    return num_batches