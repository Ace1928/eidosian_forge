import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def alibi_pattern(threshold: float, mask_shape: torch.Size) -> torch.Tensor:
    """
    Use the additive bias computation from ALiBi_ to generate a mask.
    Note that this mask can in turn be used to generate a blocksparse attention computation layout

    .. note: mask_shape is expected to hold the [heads, seq, seq] dimensions

    .. _ALiBi: https://arxiv.org/pdf/2108.12409.pdf
    """

    def get_slopes(n: int):

        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-2 ** (-(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
    maxpos = mask_shape[1]
    attn_heads = mask_shape[0]
    slopes = torch.Tensor(get_slopes(attn_heads))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
    alibi = alibi.view(attn_heads, 1, maxpos)
    return alibi < threshold