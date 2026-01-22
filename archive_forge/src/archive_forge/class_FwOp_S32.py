from typing import Any, List, Optional, Set, Tuple
import torch
from xformers.ops.common import get_xformers_operator, register_operator
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from xformers.ops.fmha.common import (
class FwOp_S32(FwOp):
    SPLIT_K = 32
    NAME = 'ck_splitK32'