from typing import Literal
import torch
from torch import Tensor
from torchmetrics.functional.clustering.mutual_info_score import mutual_info_score
from torchmetrics.functional.clustering.utils import (
Compute normalized mutual information between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        average_method: normalizer computation method

    Returns:
        Scalar tensor with normalized mutual info score between 0.0 and 1.0

    Example:
        >>> from torchmetrics.functional.clustering import normalized_mutual_info_score
        >>> target = torch.tensor([0, 3, 2, 2, 1])
        >>> preds = torch.tensor([1, 3, 2, 0, 1])
        >>> normalized_mutual_info_score(preds, target, "arithmetic")
        tensor(0.7919)

    