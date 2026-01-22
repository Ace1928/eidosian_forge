import torch
from torch import Tensor, tensor
from torchmetrics.functional.clustering.utils import calculate_contingency_matrix, check_cluster_labels
def _mutual_info_score_update(preds: Tensor, target: Tensor) -> Tensor:
    """Update and return variables required to compute the mutual information score.

    Args:
        preds: predicted class labels
        target: ground truth class labels

    Returns:
        contingency: contingency matrix

    """
    check_cluster_labels(preds, target)
    return calculate_contingency_matrix(preds, target)