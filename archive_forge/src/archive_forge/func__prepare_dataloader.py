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
def _prepare_dataloader(self, dataloader: object, shuffle: bool, mode: RunningStage) -> object:
    """This function handles the following functionalities:

        - Injecting a `DistributedDataSamplerWrapper` into the `DataLoader` if on a distributed environment
        - Wrapping the dataloader based on strategy-specific logic

        """
    if not isinstance(dataloader, DataLoader):
        return dataloader
    if _graphcore_available_and_importable():
        from lightning_graphcore import IPUAccelerator
        is_ipu = isinstance(self.trainer.accelerator, IPUAccelerator)
    else:
        is_ipu = False
    if self._requires_distributed_sampler(dataloader) or mode == RunningStage.PREDICTING or is_ipu:
        sampler = self._resolve_sampler(dataloader, shuffle=shuffle, mode=mode)
        return _update_dataloader(dataloader, sampler, mode=mode)
    return dataloader