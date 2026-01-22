from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _stat_scores(preds: Tensor, target: Tensor, reduce: Optional[str]='micro') -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate the number of tp, fp, tn, fn.

    Args:
        preds: An ``(N, C)`` or ``(N, C, X)`` tensor of predictions (0 or 1)
        target: An ``(N, C)`` or ``(N, C, X)`` tensor of true labels (0 or 1)
        reduce: One of ``'micro'``, ``'macro'``, ``'samples'``

    Return:
        Returns a list of 4 tensors; tp, fp, tn, fn.
        The shape of the returned tensors depends on the shape of the inputs
        and the ``reduce`` parameter:

        If inputs are of the shape ``(N, C)``, then:

        - If ``reduce='micro'``, the returned tensors are 1 element tensors
        - If ``reduce='macro'``, the returned tensors are ``(C,)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,)`` tensors

        If inputs are of the shape ``(N, C, X)``, then:

        - If ``reduce='micro'``, the returned tensors are ``(N,)`` tensors
        - If ``reduce='macro'``, the returned tensors are ``(N,C)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,X)`` tensors

    """
    dim: Union[int, List[int]] = 1
    if reduce == 'micro':
        dim = [0, 1] if preds.ndim == 2 else [1, 2]
    elif reduce == 'macro':
        dim = 0 if preds.ndim == 2 else 2
    true_pred, false_pred = (target == preds, target != preds)
    pos_pred, neg_pred = (preds == 1, preds == 0)
    tp = (true_pred * pos_pred).sum(dim=dim)
    fp = (false_pred * pos_pred).sum(dim=dim)
    tn = (true_pred * neg_pred).sum(dim=dim)
    fn = (false_pred * neg_pred).sum(dim=dim)
    return (tp.long(), fp.long(), tn.long(), fn.long())