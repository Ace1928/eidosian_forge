from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn, reduce
def _psnr_compute(sum_squared_error: Tensor, num_obs: Tensor, data_range: Tensor, base: float=10.0, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean') -> Tensor:
    """Compute peak signal-to-noise ratio.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        num_obs: Number of predictions or observations
        data_range: the range of the data. If None, it is determined from the data (max - min).
           ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Example:
        >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> data_range = target.max() - target.min()
        >>> sum_squared_error, num_obs = _psnr_update(preds, target)
        >>> _psnr_compute(sum_squared_error, num_obs, data_range)
        tensor(2.5527)

    """
    psnr_base_e = 2 * torch.log(data_range) - torch.log(sum_squared_error / num_obs)
    psnr_vals = psnr_base_e * (10 / torch.log(tensor(base)))
    return reduce(psnr_vals, reduction=reduction)