from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..swin_transformer import PatchMerging, SwinTransformerBlock
def _compute_attention_mask_3d(x: Tensor, size_dhw: Tuple[int, int, int], window_size: Tuple[int, int, int], shift_size: Tuple[int, int, int]) -> Tensor:
    attn_mask = x.new_zeros(*size_dhw)
    num_windows = size_dhw[0] // window_size[0] * (size_dhw[1] // window_size[1]) * (size_dhw[2] // window_size[2])
    slices = [((0, -window_size[i]), (-window_size[i], -shift_size[i]), (-shift_size[i], None)) for i in range(3)]
    count = 0
    for d in slices[0]:
        for h in slices[1]:
            for w in slices[2]:
                attn_mask[d[0]:d[1], h[0]:h[1], w[0]:w[1]] = count
                count += 1
    attn_mask = attn_mask.view(size_dhw[0] // window_size[0], window_size[0], size_dhw[1] // window_size[1], window_size[1], size_dhw[2] // window_size[2], window_size[2])
    attn_mask = attn_mask.permute(0, 2, 4, 1, 3, 5).reshape(num_windows, window_size[0] * window_size[1] * window_size[2])
    attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask