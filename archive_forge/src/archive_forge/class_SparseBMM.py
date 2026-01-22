import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union
import torch
from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask
class SparseBMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        a = a.coalesce()
        r = torch.bmm(a, b)
        ctx.save_for_backward(a, b)
        return r

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        ga = None
        if ctx.needs_input_grad[0]:
            ga = torch.ops.xformers.matmul_with_mask(grad, b.transpose(-2, -1), a)
        gb = None
        if ctx.needs_input_grad[1]:
            gb = a.transpose(1, 2).bmm(grad)
        return (ga, gb)