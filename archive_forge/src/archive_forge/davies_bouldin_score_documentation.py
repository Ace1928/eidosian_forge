import torch
from torch import Tensor
from torchmetrics.functional.clustering.utils import (
Compute the Davies bouldin score for clustering algorithms.

    Args:
        data: float tensor with shape ``(N,d)`` with the embedded data.
        labels: single integer tensor with shape ``(N,)`` with cluster labels

    Returns:
        Scalar tensor with the Davies bouldin score

    Example:
        >>> import torch
        >>> from torchmetrics.functional.clustering import davies_bouldin_score
        >>> _ = torch.manual_seed(42)
        >>> data = torch.randn(10, 3)
        >>> labels = torch.randint(0, 2, (10,))
        >>> davies_bouldin_score(data, labels)
        tensor(1.3249)

    