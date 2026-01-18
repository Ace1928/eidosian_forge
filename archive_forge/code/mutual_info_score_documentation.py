import torch
from torch import Tensor, tensor
from torchmetrics.functional.clustering.utils import calculate_contingency_matrix, check_cluster_labels
Compute mutual information between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels

    Example:
        >>> from torchmetrics.functional.clustering import mutual_info_score
        >>> target = torch.tensor([0, 3, 2, 2, 1])
        >>> preds = torch.tensor([1, 3, 2, 0, 1])
        >>> mutual_info_score(preds, target)
        tensor(1.0549)

    