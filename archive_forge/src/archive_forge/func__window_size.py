import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
def _window_size(attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> Tuple[int, int]:
    win_left = -1
    win_right = -1
    if isinstance(attn_bias, (BlockDiagonalCausalLocalAttentionMask, BlockDiagonalCausalLocalAttentionFromBottomRightMask, LowerTriangularFromBottomRightLocalAttentionMask)):
        win_left = attn_bias._window_size - 1
    if isinstance(attn_bias, LocalAttentionFromBottomRightMask):
        win_left = attn_bias.window_left
        win_right = attn_bias.window_right
    return (win_left, win_right)