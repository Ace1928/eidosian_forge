from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torchmetrics import Metric
from typing_extensions import TypedDict, override
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning_fabric.utilities.distributed import _distributed_is_initialized
from lightning_fabric.utilities.imports import _TORCH_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.data import extract_batch_size
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_1_0_0
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
from pytorch_lightning.utilities.warnings import PossibleUserWarning
def _forked_name(self, result_metric: _ResultMetric, on_step: bool) -> Tuple[str, str]:
    name = result_metric.meta.name
    forked_name = result_metric.meta.forked_name(on_step)
    add_dataloader_idx = result_metric.meta.add_dataloader_idx
    dl_idx = result_metric.meta.dataloader_idx
    if add_dataloader_idx and dl_idx is not None:
        dataloader_suffix = self.DATALOADER_SUFFIX.format(dl_idx)
        name += dataloader_suffix
        forked_name += dataloader_suffix
    return (name, forked_name)