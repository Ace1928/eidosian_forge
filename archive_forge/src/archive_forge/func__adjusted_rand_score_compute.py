import torch
from torch import Tensor
from torchmetrics.functional.clustering.utils import (
def _adjusted_rand_score_compute(contingency: Tensor) -> Tensor:
    """Compute the rand score based on the contingency matrix.

    Args:
        contingency: contingency matrix

    Returns:
        rand_score: rand score

    """
    (tn, fp), (fn, tp) = calculate_pair_cluster_confusion_matrix(contingency=contingency)
    if fn == 0 and fp == 0:
        return torch.ones_like(tn, dtype=torch.float32)
    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))