from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation
from . import _utils as det_utils
from .anchor_utils import AnchorGenerator  # noqa: 401
from .image_list import ImageList
def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
    r = []
    offset = 0
    for ob in objectness.split(num_anchors_per_level, 1):
        num_anchors = ob.shape[1]
        pre_nms_top_n = det_utils._topk_min(ob, self.pre_nms_top_n(), 1)
        _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
        r.append(top_n_idx + offset)
        offset += num_anchors
    return torch.cat(r, dim=1)