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
class _SpectralAngleMapper(SpectralAngleMapper):
    """Wrapper for deprecated import.

    >>> import torch
    >>> gen = torch.manual_seed(42)
    >>> preds = torch.rand([16, 3, 16, 16], generator=gen)
    >>> target = torch.rand([16, 3, 16, 16], generator=gen)
    >>> sam = _SpectralAngleMapper()
    >>> sam(preds, target)
    tensor(0.5914)

    """

    def __init__(self, reduction: Literal['elementwise_mean', 'sum', 'none']='elementwise_mean', **kwargs: Any) -> None:
        _deprecated_root_import_class('SpectralAngleMapper', 'image')
        super().__init__(reduction=reduction, **kwargs)