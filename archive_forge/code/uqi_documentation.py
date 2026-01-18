from typing import Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.functional.image.utils import _gaussian_kernel_2d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
Universal Image Quality Index.

    Args:
        preds: estimated image
        target: ground truth image
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor with UniversalImageQualityIndex score

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
        >>> from torchmetrics.functional.image import universal_image_quality_index
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> universal_image_quality_index(preds, target)
        tensor(0.9216)

    References:
        [1] Zhou Wang and A. C. Bovik, "A universal image quality index," in IEEE Signal Processing Letters, vol. 9,
        no. 3, pp. 81-84, March 2002, doi: 10.1109/97.995823.

        [2] Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: from error visibility
        to structural similarity," in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, April 2004,
        doi: 10.1109/TIP.2003.819861.

    