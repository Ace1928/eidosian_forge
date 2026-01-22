import math
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d, pad
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _scc_update(preds: Tensor, target: Tensor, hp_filter: Tensor, window_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Update and returns variables required to compute Spatial Correlation Coefficient.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        hp_filter: High-pass filter tensor
        window_size: Local window size integer

    Return:
        Tuple of (preds, target, hp_filter) tensors

    Raises:
        ValueError:
            If ``preds`` and ``target`` have different number of channels
            If ``preds`` and ``target`` have different shapes
            If ``preds`` and ``target`` have invalid shapes
            If ``window_size`` is not a positive integer
            If ``window_size`` is greater than the size of the image

    """
    if preds.dtype != target.dtype:
        target = target.to(preds.dtype)
    _check_same_shape(preds, target)
    if preds.ndim not in (3, 4):
        raise ValueError(f'Expected `preds` and `target` to have batch of colored images with BxCxHxW shape  or batch of grayscale images of BxHxW shape. Got preds: {preds.shape} and target: {target.shape}.')
    if len(preds.shape) == 3:
        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)
    if not window_size > 0:
        raise ValueError(f'Expected `window_size` to be a positive integer. Got {window_size}.')
    if window_size > preds.size(2) or window_size > preds.size(3):
        raise ValueError(f'Expected `window_size` to be less than or equal to the size of the image. Got window_size: {window_size} and image size: {preds.size(2)}x{preds.size(3)}.')
    preds = preds.to(torch.float32)
    target = target.to(torch.float32)
    hp_filter = hp_filter[None, None, :].to(dtype=preds.dtype, device=preds.device)
    return (preds, target, hp_filter)