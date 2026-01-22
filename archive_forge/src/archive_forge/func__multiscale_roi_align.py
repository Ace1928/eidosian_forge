from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.fx
import torchvision
from torch import nn, Tensor
from torchvision.ops.boxes import box_area
from ..utils import _log_api_usage_once
from .roi_align import roi_align
@torch.fx.wrap
def _multiscale_roi_align(x_filtered: List[Tensor], boxes: List[Tensor], output_size: List[int], sampling_ratio: int, scales: Optional[List[float]], mapper: Optional[LevelMapper]) -> Tensor:
    """
    Args:
        x_filtered (List[Tensor]): List of input tensors.
        boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
            (x1, y1, x2, y2) format and in the image reference size, not the feature map
            reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        output_size (Union[List[Tuple[int, int]], List[int]]): size of the output
        sampling_ratio (int): sampling ratio for ROIAlign
        scales (Optional[List[float]]): If None, scales will be automatically inferred. Default value is None.
        mapper (Optional[LevelMapper]): If none, mapper will be automatically inferred. Default value is None.
    Returns:
        result (Tensor)
    """
    if scales is None or mapper is None:
        raise ValueError('scales and mapper should not be None')
    num_levels = len(x_filtered)
    rois = _convert_to_roi_format(boxes)
    if num_levels == 1:
        return roi_align(x_filtered[0], rois, output_size=output_size, spatial_scale=scales[0], sampling_ratio=sampling_ratio)
    levels = mapper(boxes)
    num_rois = len(rois)
    num_channels = x_filtered[0].shape[1]
    dtype, device = (x_filtered[0].dtype, x_filtered[0].device)
    result = torch.zeros((num_rois, num_channels) + output_size, dtype=dtype, device=device)
    tracing_results = []
    for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
        idx_in_level = torch.where(levels == level)[0]
        rois_per_level = rois[idx_in_level]
        result_idx_in_level = roi_align(per_level_feature, rois_per_level, output_size=output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
        if torchvision._is_tracing():
            tracing_results.append(result_idx_in_level.to(dtype))
        else:
            result[idx_in_level] = result_idx_in_level.to(result.dtype)
    if torchvision._is_tracing():
        result = _onnx_merge_levels(levels, tracing_results)
    return result