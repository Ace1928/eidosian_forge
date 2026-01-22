from typing import Optional, Tuple
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide
def _critical_success_index_update(preds: Tensor, target: Tensor, threshold: float, keep_sequence_dim: Optional[int]=None) -> Tuple[Tensor, Tensor, Tensor]:
    """Update and return variables required to compute Critical Success Index. Checks for same shape of tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Values above or equal to threshold are replaced with 1, below by 0
        keep_sequence_dim: Index of the sequence dimension if the inputs are sequences of images. If specified,
            the score will be calculated separately for each image in the sequence. If ``None``, the score will be
            calculated across all dimensions.

    """
    _check_same_shape(preds, target)
    if keep_sequence_dim is None:
        sum_dims = None
    elif not 0 <= keep_sequence_dim < preds.ndim:
        raise ValueError(f'Expected keep_sequence dim to be in range [0, {preds.ndim}] but got {keep_sequence_dim}')
    else:
        sum_dims = tuple((i for i in range(preds.ndim) if i != keep_sequence_dim))
    preds_bin = (preds >= threshold).bool()
    target_bin = (target >= threshold).bool()
    if keep_sequence_dim is None:
        hits = torch.sum(preds_bin & target_bin).int()
        misses = torch.sum((preds_bin ^ target_bin) & target_bin).int()
        false_alarms = torch.sum((preds_bin ^ target_bin) & preds_bin).int()
    else:
        hits = torch.sum(preds_bin & target_bin, dim=sum_dims).int()
        misses = torch.sum((preds_bin ^ target_bin) & target_bin, dim=sum_dims).int()
        false_alarms = torch.sum((preds_bin ^ target_bin) & preds_bin, dim=sum_dims).int()
    return (hits, misses, false_alarms)