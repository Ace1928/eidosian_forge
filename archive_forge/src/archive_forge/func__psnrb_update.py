import math
from typing import Tuple
import torch
from torch import Tensor, tensor
def _psnrb_update(preds: Tensor, target: Tensor, block_size: int=8) -> Tuple[Tensor, Tensor, Tensor]:
    """Updates and returns variables required to compute peak signal-to-noise ratio.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        block_size: Integer indication the block size

    """
    sum_squared_error = torch.sum(torch.pow(preds - target, 2))
    num_obs = tensor(target.numel(), device=target.device)
    bef = _compute_bef(preds, block_size=block_size)
    return (sum_squared_error, bef, num_obs)