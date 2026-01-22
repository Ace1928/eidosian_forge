from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
def _get_seqlen_info(inp: Inputs) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    attn_bias = inp.attn_bias
    if isinstance(attn_bias, (BlockDiagonalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask)):
        seqstart_k = attn_bias.k_seqinfo.seqstart
        seqstart_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
    else:
        seqstart_k = None
        seqstart_q = None
        max_seqlen_q = -1
    return (seqstart_k, seqstart_q, max_seqlen_q)