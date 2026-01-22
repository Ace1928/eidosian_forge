from typing import Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _ergas_compute(preds: Tensor, target: Tensor, ratio: float=4, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean') -> Tensor:
    """Erreur Relative Globale Adimensionnelle de Synthèse.

    Args:
        preds: estimated image
        target: ground truth image
        ratio: ratio of high resolution to low resolution
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Example:
        >>> gen = torch.manual_seed(42)
        >>> preds = torch.rand([16, 1, 16, 16], generator=gen)
        >>> target = preds * 0.75
        >>> preds, target = _ergas_update(preds, target)
        >>> torch.round(_ergas_compute(preds, target))
        tensor(154.)

    """
    b, c, h, w = preds.shape
    preds = preds.reshape(b, c, h * w)
    target = target.reshape(b, c, h * w)
    diff = preds - target
    sum_squared_error = torch.sum(diff * diff, dim=2)
    rmse_per_band = torch.sqrt(sum_squared_error / (h * w))
    mean_target = torch.mean(target, dim=2)
    ergas_score = 100 * ratio * torch.sqrt(torch.sum((rmse_per_band / mean_target) ** 2, dim=1) / c)
    return reduce(ergas_score, reduction)