from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.functional.image.utils import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _multiscale_ssim_compute(mcs_per_image: Tensor, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean') -> Tensor:
    """Apply the specified reduction to pre-computed multi-scale structural similarity.

    Args:
        mcs_per_image: per image similarities for a batch of images.
        reduction: a method to reduce metric score over individual batch scores

                - ``'elementwise_mean'``: takes the mean
                - ``'sum'``: takes the sum
                - ``'none'`` or ``None``: no reduction will be applied

    Returns:
        The reduced multi-scale structural similarity

    """
    return reduce(mcs_per_image, reduction)