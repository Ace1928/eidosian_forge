from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.fx
import torchvision
from torch import nn, Tensor
from torchvision.ops.boxes import box_area
from ..utils import _log_api_usage_once
from .roi_align import roi_align
def _infer_scale(feature: Tensor, original_size: List[int]) -> float:
    size = feature.shape[-2:]
    possible_scales: List[float] = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** float(torch.tensor(approx_scale).log2().round())
        possible_scales.append(scale)
    return possible_scales[0]