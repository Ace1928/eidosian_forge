import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
def _post_process_lse(lse: torch.Tensor, inp: Inputs, original_query_shape: Tuple[int, ...]) -> torch.Tensor:
    if not isinstance(inp.attn_bias, (BlockDiagonalGappyKeysMask, BlockDiagonalPaddedKeysMask)):
        if inp.is_partial and len(original_query_shape) == 5:
            return lse.unflatten(1, original_query_shape[2:4])
        return lse
    q_seqinfo = inp.attn_bias.q_seqinfo
    B = len(q_seqinfo.seqstart_py) - 1
    if q_seqinfo.max_seqlen * B != original_query_shape[1]:
        return lse
    lse_hkm = lse.permute(1, 0, 2).flatten(start_dim=1)[None]
    if len(original_query_shape) == 5:
        return lse_hkm.unflatten(1, original_query_shape[2:4])
    return lse_hkm