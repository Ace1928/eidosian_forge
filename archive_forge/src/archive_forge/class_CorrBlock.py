from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
class CorrBlock(nn.Module):
    """The correlation block.

    Creates a correlation pyramid with ``num_levels`` levels from the outputs of the feature encoder,
    and then indexes from this pyramid to create correlation features.
    The "indexing" of a given centroid pixel x' is done by concatenating its surrounding neighbors that
    are within a ``radius``, according to the infinity norm (see paper section 3.2).
    Note: typo in the paper, it should be infinity norm, not 1-norm.
    """

    def __init__(self, *, num_levels: int=4, radius: int=4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid: List[Tensor] = [torch.tensor(0)]
        self.out_channels = num_levels * (2 * radius + 1) ** 2

    def build_pyramid(self, fmap1, fmap2):
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2)
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """
        if fmap1.shape != fmap2.shape:
            raise ValueError(f'Input feature maps should have the same shape, instead got {fmap1.shape} (fmap1.shape) != {fmap2.shape} (fmap2.shape)')
        min_fmap_size = 2 * 2 ** (self.num_levels - 1)
        if any((fmap_size < min_fmap_size for fmap_size in fmap1.shape[-2:])):
            raise ValueError(f'Feature maps are too small to be down-sampled by the correlation pyramid. H and W of feature maps should be at least {min_fmap_size}; got: {fmap1.shape[-2:]}. Remember that input images to the model are downsampled by 8, so that means their dimensions should be at least 8 * {min_fmap_size} = {8 * min_fmap_size}.')
        corr_volume = self._compute_corr_volume(fmap1, fmap2)
        batch_size, h, w, num_channels, _, _ = corr_volume.shape
        corr_volume = corr_volume.reshape(batch_size * h * w, num_channels, h, w)
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_volume)

    def index_pyramid(self, centroids_coords):
        """Return correlation features by indexing from the pyramid."""
        neighborhood_side_len = 2 * self.radius + 1
        di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        delta = torch.stack(torch.meshgrid(di, dj, indexing='ij'), dim=-1).to(centroids_coords.device)
        delta = delta.view(1, neighborhood_side_len, neighborhood_side_len, 2)
        batch_size, _, h, w = centroids_coords.shape
        centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)
        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:
            sampling_coords = centroids_coords + delta
            indexed_corr_volume = grid_sample(corr_volume, sampling_coords, align_corners=True, mode='bilinear').view(batch_size, h, w, -1)
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords = centroids_coords / 2
        corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()
        expected_output_shape = (batch_size, self.out_channels, h, w)
        if corr_features.shape != expected_output_shape:
            raise ValueError(f'Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}')
        return corr_features

    def _compute_corr_volume(self, fmap1, fmap2):
        batch_size, num_channels, h, w = fmap1.shape
        fmap1 = fmap1.view(batch_size, num_channels, h * w)
        fmap2 = fmap2.view(batch_size, num_channels, h * w)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch_size, h, w, 1, h, w)
        return corr / torch.sqrt(torch.tensor(num_channels))