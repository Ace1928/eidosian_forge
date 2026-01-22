from typing import Optional, Sequence, Tuple, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.d_lambda import spectral_distortion_index
from torchmetrics.functional.image.ergas import error_relative_global_dimensionless_synthesis
from torchmetrics.functional.image.gradients import image_gradients
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.rase import relative_average_spectral_error
from torchmetrics.functional.image.rmse_sw import root_mean_squared_error_using_sliding_window
from torchmetrics.functional.image.sam import spectral_angle_mapper
from torchmetrics.functional.image.ssim import (
from torchmetrics.functional.image.tv import total_variation
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.prints import _deprecated_root_import_func
def _multiscale_structural_similarity_index_measure(preds: Tensor, target: Tensor, gaussian_kernel: bool=True, sigma: Union[float, Sequence[float]]=1.5, kernel_size: Union[int, Sequence[int]]=11, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean', data_range: Optional[Union[float, Tuple[float, float]]]=None, k1: float=0.01, k2: float=0.03, betas: Tuple[float, ...]=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), normalize: Optional[Literal['relu', 'simple']]='relu') -> Tensor:
    """Wrapper for deprecated import.

    >>> import torch
    >>> gen = torch.manual_seed(42)
    >>> preds = torch.rand([3, 3, 256, 256], generator=gen)
    >>> target = preds * 0.75
    >>> _multiscale_structural_similarity_index_measure(preds, target, data_range=1.0)
    tensor(0.9627)

    """
    _deprecated_root_import_func('multiscale_structural_similarity_index_measure', 'image')
    return multiscale_structural_similarity_index_measure(preds=preds, target=target, gaussian_kernel=gaussian_kernel, sigma=sigma, kernel_size=kernel_size, reduction=reduction, data_range=data_range, k1=k1, k2=k2, betas=betas, normalize=normalize)