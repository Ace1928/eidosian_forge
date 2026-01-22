from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities.checks import _check_same_shape
def _spearman_corrcoef_compute(preds: Tensor, target: Tensor, eps: float=1e-06) -> Tensor:
    """Compute Spearman Correlation Coefficient.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        eps: Avoids ``ZeroDivisionError``.

    Example:
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> preds, target = _spearman_corrcoef_update(preds, target, num_outputs=1)
        >>> _spearman_corrcoef_compute(preds, target)
        tensor(1.0000)

    """
    if preds.ndim == 1:
        preds = _rank_data(preds)
        target = _rank_data(target)
    else:
        preds = torch.stack([_rank_data(p) for p in preds.T]).T
        target = torch.stack([_rank_data(t) for t in target.T]).T
    preds_diff = preds - preds.mean(0)
    target_diff = target - target.mean(0)
    cov = (preds_diff * target_diff).mean(0)
    preds_std = torch.sqrt((preds_diff * preds_diff).mean(0))
    target_std = torch.sqrt((target_diff * target_diff).mean(0))
    corrcoef = cov / (preds_std * target_std + eps)
    return torch.clamp(corrcoef, -1.0, 1.0)