from typing import Optional
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_retrieval_functional_inputs
Compute the hit rate for information retrieval.

    The hit rate is 1.0 if there is at least one relevant document among all the top `k` retrieved documents.

    ``preds`` and ``target`` should be of the same shape and live on the same device. If no ``target`` is ``True``,
    ``0`` is returned. ``target`` must be either `bool` or `integers` and ``preds`` must be ``float``,
    otherwise an error is raised. If you want to measure HitRate@K, ``top_k`` must be a positive integer.

    Args:
        preds: estimated probabilities of each document to be relevant.
        target: ground truth about each document being relevant or not.
        top_k: consider only the top k elements (default: `None`, which considers them all)

    Returns:
        A single-value tensor with the hit rate (at ``top_k``) of the predictions ``preds`` w.r.t. the labels
          ``target``.

    Raises:
        ValueError:
            If ``top_k`` parameter is not `None` or an integer larger than 0

    Example:
        >>> from torch import tensor
        >>> preds = tensor([0.2, 0.3, 0.5])
        >>> target = tensor([True, False, True])
        >>> retrieval_hit_rate(preds, target, top_k=2)
        tensor(1.)

    