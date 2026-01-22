from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.data import _cumsum
def _multilabel_ranking_average_precision_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Accumulate state for label ranking average precision."""
    neg_preds = -preds
    score = torch.tensor(0.0, device=neg_preds.device)
    num_preds, num_labels = neg_preds.shape
    for i in range(num_preds):
        relevant = target[i] == 1
        ranking = _rank_data(neg_preds[i][relevant]).float()
        if len(ranking) > 0 and len(ranking) < num_labels:
            rank = _rank_data(neg_preds[i])[relevant].float()
            score_idx = (ranking / rank).mean()
        else:
            score_idx = torch.ones_like(score)
        score += score_idx
    return (score, num_preds)