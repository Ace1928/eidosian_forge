from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.fx
import torchvision
from torch import nn, Tensor
from torchvision.ops.boxes import box_area
from ..utils import _log_api_usage_once
from .roi_align import roi_align
class LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(self, k_min: int, k_max: int, canonical_scale: int=224, canonical_level: int=4, eps: float=1e-06):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        Args:
            boxlists (list[BoxList])
        """
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)