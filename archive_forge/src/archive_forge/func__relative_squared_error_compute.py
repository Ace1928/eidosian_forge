from typing import Union
import torch
from torch import Tensor
from torchmetrics.functional.regression.r2 import _r2_score_update
def _relative_squared_error_compute(sum_squared_obs: Tensor, sum_obs: Tensor, sum_squared_error: Tensor, num_obs: Union[int, Tensor], squared: bool=True) -> Tensor:
    """Computes Relative Squared Error.

    Args:
        sum_squared_obs: Sum of square of all observations
        sum_obs: Sum of all observations
        sum_squared_error: Residual sum of squares
        num_obs: Number of predictions or observations
        squared: Returns RRSE value if set to False.

    Example:
        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> # RSE uses the same update function as R2 score.
        >>> sum_squared_obs, sum_obs, rss, num_obs = _r2_score_update(preds, target)
        >>> _relative_squared_error_compute(sum_squared_obs, sum_obs, rss, num_obs, squared=True)
        tensor(0.0632)

    """
    epsilon = torch.finfo(sum_squared_error.dtype).eps
    rse = sum_squared_error / torch.clamp(sum_squared_obs - sum_obs * sum_obs / num_obs, min=epsilon)
    if not squared:
        rse = torch.sqrt(rse)
    return torch.mean(rse)