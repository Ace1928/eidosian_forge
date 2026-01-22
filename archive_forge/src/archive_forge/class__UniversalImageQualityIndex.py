from typing import Any, Dict, Optional, Sequence, Tuple, Union
from typing_extensions import Literal
from torchmetrics.image.d_lambda import SpectralDistortionIndex
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.rase import RelativeAverageSpectralError
from torchmetrics.image.rmse_sw import RootMeanSquaredErrorUsingSlidingWindow
from torchmetrics.image.sam import SpectralAngleMapper
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure
from torchmetrics.image.tv import TotalVariation
from torchmetrics.image.uqi import UniversalImageQualityIndex
from torchmetrics.utilities.prints import _deprecated_root_import_class
class _UniversalImageQualityIndex(UniversalImageQualityIndex):
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand([16, 1, 16, 16])
    >>> target = preds * 0.75
    >>> uqi = _UniversalImageQualityIndex()
    >>> uqi(preds, target)
    tensor(0.9216)

    """

    def __init__(self, kernel_size: Sequence[int]=(11, 11), sigma: Sequence[float]=(1.5, 1.5), reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean', **kwargs: Any) -> None:
        _deprecated_root_import_class('UniversalImageQualityIndex', 'image')
        super().__init__(kernel_size=kernel_size, sigma=sigma, reduction=reduction, **kwargs)