import csv
import urllib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _rescale_metrics_with_baseline(precision: Tensor, recall: Tensor, f1_score: Tensor, baseline: Tensor, num_layers: Optional[int]=None, all_layers: bool=False) -> Tuple[Tensor, Tensor, Tensor]:
    """Rescale the computed metrics with the pre-computed baseline."""
    if num_layers is None and all_layers is False:
        num_layers = -1
    all_metrics = torch.stack([precision, recall, f1_score], dim=-1)
    baseline_scale = baseline.unsqueeze(1) if all_layers else baseline[num_layers]
    all_metrics = (all_metrics - baseline_scale) / (1 - baseline_scale)
    return (all_metrics[..., 0], all_metrics[..., 1], all_metrics[..., 2])