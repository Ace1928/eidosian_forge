import torch
import torch.fx
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once
from ._utils import check_roi_boxes_shape, convert_boxes_to_roi_format
class PSRoIAlign(nn.Module):
    """
    See :func:`ps_roi_align`.
    """

    def __init__(self, output_size: int, spatial_scale: float, sampling_ratio: int):
        super().__init__()
        _log_api_usage_once(self)
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return ps_roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio)

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}(output_size={self.output_size}, spatial_scale={self.spatial_scale}, sampling_ratio={self.sampling_ratio})'
        return s