import math
from typing import Literal, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torchmetrics.functional.image.lpips import _LPIPS
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE
def _perceptual_path_length_validate_arguments(num_samples: int=10000, conditional: bool=False, batch_size: int=128, interpolation_method: Literal['lerp', 'slerp_any', 'slerp_unit']='lerp', epsilon: float=0.0001, resize: Optional[int]=64, lower_discard: Optional[float]=0.01, upper_discard: Optional[float]=0.99) -> None:
    """Validate arguments for perceptual path length."""
    if not (isinstance(num_samples, int) and num_samples > 0):
        raise ValueError(f'Argument `num_samples` must be a positive integer, but got {num_samples}.')
    if not isinstance(conditional, bool):
        raise ValueError(f'Argument `conditional` must be a boolean, but got {conditional}.')
    if not (isinstance(batch_size, int) and batch_size > 0):
        raise ValueError(f'Argument `batch_size` must be a positive integer, but got {batch_size}.')
    if interpolation_method not in ['lerp', 'slerp_any', 'slerp_unit']:
        raise ValueError(f"Argument `interpolation_method` must be one of 'lerp', 'slerp_any', 'slerp_unit',got {interpolation_method}.")
    if not (isinstance(epsilon, float) and epsilon > 0):
        raise ValueError(f'Argument `epsilon` must be a positive float, but got {epsilon}.')
    if resize is not None and (not (isinstance(resize, int) and resize > 0)):
        raise ValueError(f'Argument `resize` must be a positive integer or `None`, but got {resize}.')
    if lower_discard is not None and (not (isinstance(lower_discard, float) and 0 <= lower_discard <= 1)):
        raise ValueError(f'Argument `lower_discard` must be a float between 0 and 1 or `None`, but got {lower_discard}.')
    if upper_discard is not None and (not (isinstance(upper_discard, float) and 0 <= upper_discard <= 1)):
        raise ValueError(f'Argument `upper_discard` must be a float between 0 and 1 or `None`, but got {upper_discard}.')