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
def _get_scaled_precision_or_recall(cos_sim: Tensor, metric: str, idf_scale: Tensor) -> Tensor:
    """Calculate precision or recall, transpose it and scale it with idf_scale factor."""
    dim = 3 if metric == 'precision' else 2
    res = cos_sim.max(dim=dim).values
    res = torch.einsum('bls, bs -> bls', res, idf_scale).sum(-1)
    return res.transpose(0, 1).squeeze()