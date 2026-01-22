from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTaskNoBinary
def _multiclass_exact_match_update(preds: Tensor, target: Tensor, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None) -> Tuple[Tensor, Tensor]:
    """Compute the statistics."""
    if ignore_index is not None:
        preds = preds.clone()
        preds[target == ignore_index] = ignore_index
    correct = (preds == target).sum(1) == preds.shape[1]
    correct = correct if multidim_average == 'samplewise' else correct.sum()
    total = torch.tensor(preds.shape[0] if multidim_average == 'global' else 1, device=correct.device)
    return (correct, total)