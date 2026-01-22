import math
from typing import Tuple
import torch
from torch import Tensor, tensor
def _psnrb_compute(sum_squared_error: Tensor, bef: Tensor, num_obs: Tensor, data_range: Tensor) -> Tensor:
    """Computes peak signal-to-noise ratio.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        bef: block effect
        num_obs: Number of predictions or observations
        data_range: the range of the data. If None, it is determined from the data (max - min).

    """
    sum_squared_error = sum_squared_error / num_obs + bef
    if data_range > 2:
        return 10 * torch.log10(data_range ** 2 / sum_squared_error)
    return 10 * torch.log10(1.0 / sum_squared_error)