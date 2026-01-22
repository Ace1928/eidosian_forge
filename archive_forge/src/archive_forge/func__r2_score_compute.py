from typing import Tuple, Union
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
def _r2_score_compute(sum_squared_obs: Tensor, sum_obs: Tensor, rss: Tensor, num_obs: Union[int, Tensor], adjusted: int=0, multioutput: str='uniform_average') -> Tensor:
    """Compute R2 score.

    Args:
        sum_squared_obs: Sum of square of all observations
        sum_obs: Sum of all observations
        rss: Residual sum of squares
        num_obs: Number of predictions or observations
        adjusted: number of independent regressors for calculating adjusted r2 score.
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * `'raw_values'` returns full set of scores
            * `'uniform_average'` scores are uniformly averaged
            * `'variance_weighted'` scores are weighted by their individual variances

    Example:
        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> sum_squared_obs, sum_obs, rss, num_obs = _r2_score_update(preds, target)
        >>> _r2_score_compute(sum_squared_obs, sum_obs, rss, num_obs, multioutput="raw_values")
        tensor([0.9654, 0.9082])

    """
    if num_obs < 2:
        raise ValueError('Needs at least two samples to calculate r2 score.')
    mean_obs = sum_obs / num_obs
    tss = sum_squared_obs - sum_obs * mean_obs
    cond_rss = ~torch.isclose(rss, torch.zeros_like(rss), atol=0.0001)
    cond_tss = ~torch.isclose(tss, torch.zeros_like(tss), atol=0.0001)
    cond = cond_rss & cond_tss
    raw_scores = torch.ones_like(rss)
    raw_scores[cond] = 1 - rss[cond] / tss[cond]
    raw_scores[cond_rss & ~cond_tss] = 0.0
    if multioutput == 'raw_values':
        r2 = raw_scores
    elif multioutput == 'uniform_average':
        r2 = torch.mean(raw_scores)
    elif multioutput == 'variance_weighted':
        tss_sum = torch.sum(tss)
        r2 = torch.sum(tss / tss_sum * raw_scores)
    else:
        raise ValueError(f'Argument `multioutput` must be either `raw_values`, `uniform_average` or `variance_weighted`. Received {multioutput}.')
    if adjusted < 0 or not isinstance(adjusted, int):
        raise ValueError('`adjusted` parameter should be an integer larger or equal to 0.')
    if adjusted != 0:
        if adjusted > num_obs - 1:
            rank_zero_warn('More independent regressions than data points in adjusted r2 score. Falls back to standard r2 score.', UserWarning)
        elif adjusted == num_obs - 1:
            rank_zero_warn('Division by zero in adjusted r2 score. Falls back to standard r2 score.', UserWarning)
        else:
            return 1 - (1 - r2) * (num_obs - 1) / (num_obs - adjusted - 1)
    return r2