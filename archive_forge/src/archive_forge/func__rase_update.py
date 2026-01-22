from typing import Tuple
import torch
from torch import Tensor
from torchmetrics.functional.image.rmse_sw import _rmse_sw_compute, _rmse_sw_update
from torchmetrics.functional.image.utils import _uniform_filter
def _rase_update(preds: Tensor, target: Tensor, window_size: int, rmse_map: Tensor, target_sum: Tensor, total_images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculate the sum of RMSE map values for the batch of examples and update intermediate states.

    Args:
        preds: Deformed image
        target: Ground truth image
        window_size: Sliding window used for RMSE calculation
        rmse_map: Sum of RMSE map values over all examples
        target_sum: target...
        total_images: Total number of images

    Return:
        Intermediate state of RMSE map
        Updated total number of already processed images

    """
    _, rmse_map, total_images = _rmse_sw_update(preds, target, window_size, rmse_val_sum=None, rmse_map=rmse_map, total_images=total_images)
    target_sum += torch.sum(_uniform_filter(target, window_size) / window_size ** 2, dim=0)
    return (rmse_map, target_sum, total_images)