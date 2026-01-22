import torch
from torch import Tensor
from torchmetrics.functional.clustering.utils import (
def _rand_score_update(preds: Tensor, target: Tensor) -> Tensor:
    """Update and return variables required to compute the rand score.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target)