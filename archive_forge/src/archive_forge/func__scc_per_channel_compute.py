import math
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d, pad
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _scc_per_channel_compute(preds: Tensor, target: Tensor, hp_filter: Tensor, window_size: int) -> Tensor:
    """Computes per channel Spatial Correlation Coefficient.

    Args:
        preds: estimated image of Bx1xHxW shape.
        target: ground truth image of Bx1xHxW shape.
        hp_filter: 2D high-pass filter.
        window_size: size of window for local mean calculation.

    Return:
        Tensor with Spatial Correlation Coefficient score

    """
    dtype = preds.dtype
    device = preds.device
    window = torch.ones(size=(1, 1, window_size, window_size), dtype=dtype, device=device) / window_size ** 2
    preds_hp = _hp_2d_laplacian(preds, hp_filter)
    target_hp = _hp_2d_laplacian(target, hp_filter)
    preds_var, target_var, target_preds_cov = _local_variance_covariance(preds_hp, target_hp, window)
    preds_var[preds_var < 0] = 0
    target_var[target_var < 0] = 0
    den = torch.sqrt(target_var) * torch.sqrt(preds_var)
    idx = den == 0
    den[den == 0] = 1
    scc = target_preds_cov / den
    scc[idx] = 0
    return scc