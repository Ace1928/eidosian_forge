import textwrap
from collections import deque
from typing import List, Sequence, Type, TypeVar
import torch
from . import (
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs
def _dispatch_fw_priority_list(inp: Inputs, needs_gradient: bool) -> Sequence[Type[AttentionFwOpBase]]:
    if torch.version.cuda:
        priority_list_ops = deque([flash.FwOp, cutlass.FwOp, small_k.FwOp])
    else:
        priority_list_ops = deque([ck.FwOp])
    if not needs_gradient:
        mqa_or_gqa = inp.key.ndim > 3 and inp.key.stride(-2) == 0 and (inp.key.shape[-2] > 1)
        if not mqa_or_gqa:
            priority_list_ops.appendleft(decoder.FwOp if torch.version.cuda else ck_decoder.FwOp)
        if mqa_or_gqa and inp.query.shape[1] <= 32 and (inp.key.shape[1] >= 256):
            parallelism_BH = 0
            if inp.query.ndim == 3:
                parallelism_BH = inp.query.shape[0]
            elif inp.query.ndim == 4:
                parallelism_BH = inp.query.shape[0] * inp.query.shape[2]
            elif inp.query.ndim == 5:
                parallelism_BH = inp.query.shape[0] * inp.query.shape[2]
            if parallelism_BH > 0 and parallelism_BH < 64:
                priority_list_ops.appendleft(ck_splitk.FwOp)
                priority_list_ops.appendleft(triton_splitk.FwOp)
                if not isinstance(inp.attn_bias, attn_bias.BlockDiagonalMask):
                    priority_list_ops.remove(flash.FwOp)
                    priority_list_ops.appendleft(flash.FwOp)
    return priority_list_ops