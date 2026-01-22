from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.functional.image.utils import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _ssim_check_inputs(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Update and returns variables required to compute Structural Similarity Index Measure.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    if preds.dtype != target.dtype:
        target = target.to(preds.dtype)
    _check_same_shape(preds, target)
    if len(preds.shape) not in (4, 5):
        raise ValueError(f'Expected `preds` and `target` to have BxCxHxW or BxCxDxHxW shape. Got preds: {preds.shape} and target: {target.shape}.')
    return (preds, target)