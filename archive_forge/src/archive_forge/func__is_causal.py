import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
def _is_causal(attn_bias: Optional[Union[torch.Tensor, AttentionBias]]) -> bool:
    return isinstance(attn_bias, (LowerTriangularMask, LowerTriangularFromBottomRightMask, LowerTriangularFromBottomRightLocalAttentionMask, BlockDiagonalCausalMask, BlockDiagonalCausalLocalAttentionMask, BlockDiagonalCausalFromBottomRightMask, BlockDiagonalCausalLocalAttentionFromBottomRightMask, BlockDiagonalCausalWithOffsetGappyKeysMask, BlockDiagonalCausalWithOffsetPaddedKeysMask))