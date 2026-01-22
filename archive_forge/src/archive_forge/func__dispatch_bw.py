import textwrap
from collections import deque
from typing import List, Sequence, Type, TypeVar
import torch
from . import (
from .common import AttentionBwOpBase, AttentionFwOpBase, Inputs
def _dispatch_bw(inp: Inputs) -> Type[AttentionBwOpBase]:
    priority_list_ops: List[Type[AttentionBwOpBase]] = [flash.BwOp, cutlass.BwOp, small_k.BwOp]
    if _is_cutlassB_faster_than_flash(inp):
        priority_list_ops.remove(cutlass.BwOp)
        priority_list_ops.insert(0, cutlass.BwOp)
    return _run_priority_list('memory_efficient_attention_backward', priority_list_ops, inp)