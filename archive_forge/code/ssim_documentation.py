from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.functional.image.utils import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
Compute `MultiScaleSSIM`_, Multi-scale Structural Similarity Index Measure.

    This metric is a generalization of Structural Similarity Index Measure by incorporating image details at different
    resolution scores.

    Args:
        preds: Predictions from model of shape ``[N, C, H, W]``
        target: Ground truth values of shape ``[N, C, H, W]``
        gaussian_kernel: If true, a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel
        kernel_size: size of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
        k1: Parameter of structural similarity index measure.
        k2: Parameter of structural similarity index measure.
        betas: Exponent parameters for individual similarities and contrastive sensitivities returned by different image
            resolutions.
        normalize: When MultiScaleSSIM loss is used for training, it is desirable to use normalizes to improve the
            training stability. This `normalize` argument is out of scope of the original implementation [1], and it is
            adapted from https://github.com/jorge-pessoa/pytorch-msssim instead.

    Return:
        Tensor with Multi-Scale SSIM score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If the length of ``kernel_size`` or ``sigma`` is not ``2``.
        ValueError:
            If one of the elements of ``kernel_size`` is not an ``odd positive number``.
        ValueError:
            If one of the elements of ``sigma`` is not a ``positive number``.

    Example:
        >>> from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
        >>> gen = torch.manual_seed(42)
        >>> preds = torch.rand([3, 3, 256, 256], generator=gen)
        >>> target = preds * 0.75
        >>> multiscale_structural_similarity_index_measure(preds, target, data_range=1.0)
        tensor(0.9627)

    References:
        [1] Multi-Scale Structural Similarity For Image Quality Assessment by Zhou Wang, Eero P. Simoncelli and Alan C.
        Bovik `MultiScaleSSIM`_

    