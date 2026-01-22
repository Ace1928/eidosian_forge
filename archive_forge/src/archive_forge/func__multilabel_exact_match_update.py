from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTaskNoBinary
def _multilabel_exact_match_update(preds: Tensor, target: Tensor, num_labels: int, multidim_average: Literal['global', 'samplewise']='global') -> Tuple[Tensor, Tensor]:
    """Compute the statistics."""
    if multidim_average == 'global':
        preds = torch.movedim(preds, 1, -1).reshape(-1, num_labels)
        target = torch.movedim(target, 1, -1).reshape(-1, num_labels)
    correct = ((preds == target).sum(1) == num_labels).sum(dim=-1)
    total = torch.tensor(preds.shape[0 if multidim_average == 'global' else 2], device=correct.device)
    return (correct, total)