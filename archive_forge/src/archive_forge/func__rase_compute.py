from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.image.rmse_sw import _rmse_sw_compute, _rmse_sw_update
from torchmetrics.functional.image.utils import _uniform_filter
def _rase_compute(rmse_map: Tensor, target_sum: Tensor, total_images: Tensor, window_size: int) -> Tensor:
    """Compute RASE.

    Args:
        rmse_map: Sum of RMSE map values over all examples
        target_sum: target...
        total_images: Total number of images.
        window_size: Sliding window used for rmse calculation

    Return:
        Relative Average Spectral Error (RASE)

    """
    _, rmse_map = _rmse_sw_compute(rmse_val_sum=None, rmse_map=rmse_map, total_images=total_images)
    target_mean = target_sum / total_images
    target_mean = target_mean.mean(0)
    rase_map = 100 / target_mean * torch.sqrt(torch.mean(rmse_map ** 2, 0))
    crop_slide = round(window_size / 2)
    return torch.mean(rase_map[crop_slide:-crop_slide, crop_slide:-crop_slide])