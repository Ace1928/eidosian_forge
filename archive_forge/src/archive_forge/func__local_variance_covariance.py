import math
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d, pad
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _local_variance_covariance(preds: Tensor, target: Tensor, window: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes local variance and covariance of the input tensors."""
    left_padding = int(math.ceil((window.size(3) - 1) / 2))
    right_padding = int(math.floor((window.size(3) - 1) / 2))
    preds = pad(preds, (left_padding, right_padding, left_padding, right_padding))
    target = pad(target, (left_padding, right_padding, left_padding, right_padding))
    preds_mean = conv2d(preds, window, stride=1, padding=0)
    target_mean = conv2d(target, window, stride=1, padding=0)
    preds_var = conv2d(preds ** 2, window, stride=1, padding=0) - preds_mean ** 2
    target_var = conv2d(target ** 2, window, stride=1, padding=0) - target_mean ** 2
    target_preds_cov = conv2d(target * preds, window, stride=1, padding=0) - target_mean * preds_mean
    return (preds_var, target_var, target_preds_cov)