import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, FrozenBatchNorm2d, generalized_box_iou_loss
class SSDMatcher(Matcher):

    def __init__(self, threshold: float) -> None:
        super().__init__(threshold, threshold, allow_low_quality_matches=False)

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        matches = super().__call__(match_quality_matrix)
        _, highest_quality_pred_foreach_gt = match_quality_matrix.max(dim=1)
        matches[highest_quality_pred_foreach_gt] = torch.arange(highest_quality_pred_foreach_gt.size(0), dtype=torch.int64, device=highest_quality_pred_foreach_gt.device)
        return matches