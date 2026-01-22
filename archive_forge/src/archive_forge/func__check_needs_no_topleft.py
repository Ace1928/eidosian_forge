import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
def _check_needs_no_topleft(d: Inputs, reasons: List[str]) -> None:
    if isinstance(d.attn_bias, BlockDiagonalCausalMask):
        for k_start, q_start in zip_longest(d.attn_bias.k_seqinfo.seqstart_py, d.attn_bias.q_seqinfo.seqstart_py):
            if k_start != q_start:
                reasons.append('Only support BlockDiagonalCausalMask if equal numbers of keys and queries')
                break
    elif isinstance(d.attn_bias, LowerTriangularMask):
        if d.query.shape[1] != d.key.shape[1]:
            reasons.append('Only support LowerTriangularMask if equal number ofkeys and queries')