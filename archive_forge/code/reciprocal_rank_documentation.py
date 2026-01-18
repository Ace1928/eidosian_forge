from typing import Optional
import torch
from torch import Tensor, tensor
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
Compute reciprocal rank (for information retrieval). See `Mean Reciprocal Rank`_.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    0 is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        top_k: consider only the top k elements (default: ``None``, which considers them all)

    Return:
        a single-value tensor with the reciprocal rank (RR) of the predictions ``preds`` wrt the labels ``target``.

    Raises:
        ValueError:
            If ``top_k`` is not ``None`` or an integer larger than 0.

    Example:
        >>> from torchmetrics.functional.retrieval import retrieval_reciprocal_rank
        >>> preds = torch.tensor([0.2, 0.3, 0.5])
        >>> target = torch.tensor([False, True, False])
        >>> retrieval_reciprocal_rank(preds, target)
        tensor(0.5000)

    