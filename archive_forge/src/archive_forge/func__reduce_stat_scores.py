from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _reduce_stat_scores(numerator: Tensor, denominator: Tensor, weights: Optional[Tensor], average: Optional[str], mdmc_average: Optional[str], zero_division: int=0) -> Tensor:
    """Reduces scores of type ``numerator/denominator`` or.

    ``weights * (numerator/denominator)``, if ``average='weighted'``.

    Args:
        numerator: A tensor with numerator numbers.
        denominator: A tensor with denominator numbers. If a denominator is
            negative, the class will be ignored (if averaging), or its score
            will be returned as ``nan`` (if ``average=None``).
            If the denominator is zero, then ``zero_division`` score will be
            used for those elements.
        weights: A tensor of weights to be used if ``average='weighted'``.
        average: The method to average the scores
        mdmc_average: The method to average the scores if inputs were multi-dimensional multi-class (MDMC)
        zero_division: The value to use for the score if denominator equals zero.

    """
    numerator, denominator = (numerator.float(), denominator.float())
    zero_div_mask = denominator == 0
    ignore_mask = denominator < 0
    weights = torch.ones_like(denominator) if weights is None else weights.float()
    numerator = torch.where(zero_div_mask, tensor(zero_division, dtype=numerator.dtype, device=numerator.device), numerator)
    denominator = torch.where(zero_div_mask | ignore_mask, tensor(1.0, dtype=denominator.dtype, device=denominator.device), denominator)
    weights = torch.where(ignore_mask, tensor(0.0, dtype=weights.dtype, device=weights.device), weights)
    if average not in (AverageMethod.MICRO, AverageMethod.NONE, None):
        weights = weights / weights.sum(dim=-1, keepdim=True)
    scores = weights * (numerator / denominator)
    scores = torch.where(torch.isnan(scores), tensor(zero_division, dtype=scores.dtype, device=scores.device), scores)
    if mdmc_average == MDMCAverageMethod.SAMPLEWISE:
        scores = scores.mean(dim=0)
        ignore_mask = ignore_mask.sum(dim=0).bool()
    if average in (AverageMethod.NONE, None):
        return torch.where(ignore_mask, tensor(float('nan'), device=scores.device), scores)
    return scores.sum()