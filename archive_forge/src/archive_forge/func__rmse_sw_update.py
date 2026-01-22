from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.image.utils import _uniform_filter
from torchmetrics.utilities.checks import _check_same_shape
def _rmse_sw_update(preds: Tensor, target: Tensor, window_size: int, rmse_val_sum: Optional[Tensor], rmse_map: Optional[Tensor], total_images: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculate the sum of RMSE values and RMSE map for the batch of examples and update intermediate states.

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for rmse calculation
        rmse_val_sum: Sum of RMSE over all examples per individual channels
        rmse_map: Sum of RMSE map values over all examples
        total_images: Total number of images

    Return:
        (Optionally) Intermediate state of RMSE (using sliding window) over the accumulated examples.
        (Optionally) Intermediate state of RMSE map
        Updated total number of already processed images

    Raises:
        ValueError: If ``preds`` and ``target`` do not have the same data type.
        ValueError: If ``preds`` and ``target`` do not have ``BxCxWxH`` shape.
        ValueError: If ``round(window_size / 2)`` is greater or equal to width or height of the image.

    """
    if preds.dtype != target.dtype:
        raise TypeError(f'Expected `preds` and `target` to have the same data type. But got {preds.dtype} and {target.dtype}.')
    _check_same_shape(preds, target)
    if len(preds.shape) != 4:
        raise ValueError(f'Expected `preds` and `target` to have BxCxHxW shape. But got {preds.shape}.')
    if round(window_size / 2) >= target.shape[2] or round(window_size / 2) >= target.shape[3]:
        raise ValueError(f'Parameter `round(window_size / 2)` is expected to be smaller than {min(target.shape[2], target.shape[3])} but got {round(window_size / 2)}.')
    if total_images is not None:
        total_images += target.shape[0]
    else:
        total_images = torch.tensor(target.shape[0], device=target.device)
    error = (target - preds) ** 2
    error = _uniform_filter(error, window_size)
    _rmse_map = torch.sqrt(error)
    crop_slide = round(window_size / 2)
    if rmse_val_sum is not None:
        rmse_val = _rmse_map[:, :, crop_slide:-crop_slide, crop_slide:-crop_slide]
        rmse_val_sum += rmse_val.sum(0).mean()
    else:
        rmse_val_sum = _rmse_map[:, :, crop_slide:-crop_slide, crop_slide:-crop_slide].sum(0).mean()
    if rmse_map is not None:
        rmse_map += _rmse_map.sum(0)
    else:
        rmse_map = _rmse_map.sum(0)
    return (rmse_val_sum, rmse_map, total_images)