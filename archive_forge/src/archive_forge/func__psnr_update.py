from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn, reduce
def _psnr_update(preds: Tensor, target: Tensor, dim: Optional[Union[int, Tuple[int, ...]]]=None) -> Tuple[Tensor, Tensor]:
    """Update and return variables required to compute peak signal-to-noise ratio.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        dim: Dimensions to reduce PSNR scores over provided as either an integer or a list of integers.
            Default is None meaning scores will be reduced across all dimensions.

    """
    if dim is None:
        sum_squared_error = torch.sum(torch.pow(preds - target, 2))
        num_obs = tensor(target.numel(), device=target.device)
        return (sum_squared_error, num_obs)
    diff = preds - target
    sum_squared_error = torch.sum(diff * diff, dim=dim)
    dim_list = [dim] if isinstance(dim, int) else list(dim)
    if not dim_list:
        num_obs = tensor(target.numel(), device=target.device)
    else:
        num_obs = tensor(target.size(), device=target.device)[dim_list].prod()
        num_obs = num_obs.expand_as(sum_squared_error)
    return (sum_squared_error, num_obs)