from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.distributed import reduce
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE
def _spatial_distortion_index_compute(preds: Tensor, ms: Tensor, pan: Tensor, pan_lr: Optional[Tensor]=None, norm_order: int=1, window_size: int=7, reduction: Literal['elementwise_mean', 'sum', 'none']='elementwise_mean') -> Tensor:
    """Compute Spatial Distortion Index (SpatialDistortionIndex_).

    Args:
        preds: High resolution multispectral image.
        ms: Low resolution multispectral image.
        pan: High resolution panchromatic image.
        pan_lr: Low resolution panchromatic image.
        norm_order: Order of the norm applied on the difference.
        window_size: Window size of the filter applied to degrade the high resolution panchromatic image.
        reduction: A method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Return:
        Tensor with SpatialDistortionIndex score

    Raises:
        ValueError
            If ``window_size`` is smaller than dimension of ``ms``.

    Example:
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 32, 32])
        >>> ms = torch.rand([16, 3, 16, 16])
        >>> pan = torch.rand([16, 3, 32, 32])
        >>> preds, ms, pan, pan_lr = _spatial_distortion_index_update(preds, ms, pan)
        >>> _spatial_distortion_index_compute(preds, ms, pan, pan_lr)
        tensor(0.0090)

    """
    length = preds.shape[1]
    ms_h, ms_w = ms.shape[-2:]
    if window_size >= ms_h or window_size >= ms_w:
        raise ValueError(f'Expected `window_size` to be smaller than dimension of `ms`. Got window_size: {window_size}.')
    if pan_lr is None:
        if not _TORCHVISION_AVAILABLE:
            raise ValueError('When `pan_lr` is not provided as input to metric Spatial distortion index, torchvision should be installed. Please install with `pip install torchvision` or `pip install torchmetrics[image]`.')
        from torchvision.transforms.functional import resize
        from torchmetrics.functional.image.utils import _uniform_filter
        pan_degraded = _uniform_filter(pan, window_size=window_size)
        pan_degraded = resize(pan_degraded, size=ms.shape[-2:], antialias=False)
    else:
        pan_degraded = pan_lr
    m1 = torch.zeros(length, device=preds.device)
    m2 = torch.zeros(length, device=preds.device)
    for i in range(length):
        m1[i] = universal_image_quality_index(ms[:, i:i + 1], pan_degraded[:, i:i + 1])
        m2[i] = universal_image_quality_index(preds[:, i:i + 1], pan[:, i:i + 1])
    diff = (m1 - m2).abs() ** norm_order
    return reduce(diff, reduction) ** (1 / norm_order)