from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.clustering.mutual_info_score import mutual_info_score
from torchmetrics.functional.clustering.utils import calculate_entropy, check_cluster_labels
Compute the V-measure score between two clusterings.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        beta: weight of the harmonic mean between homogeneity and completeness

    Returns:
        scalar tensor with the rand score

    Example:
        >>> from torchmetrics.functional.clustering import v_measure_score
        >>> import torch
        >>> v_measure_score(torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0]))
        tensor(1.)
        >>> v_measure_score(torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]))
        tensor(0.8000)

    