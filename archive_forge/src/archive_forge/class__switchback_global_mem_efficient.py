from functools import partial
import torch
import torch.nn as nn
from bitsandbytes.triton.dequantize_rowwise import dequantize_rowwise
from bitsandbytes.triton.int8_matmul_mixed_dequantize import (
from bitsandbytes.triton.int8_matmul_rowwise_dequantize import (
from bitsandbytes.triton.quantize_columnwise_and_transpose import (
from bitsandbytes.triton.quantize_global import (
from bitsandbytes.triton.quantize_rowwise import quantize_rowwise
from bitsandbytes.triton.triton_utils import is_triton_available
class _switchback_global_mem_efficient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_3D, W, bias):
        X = X_3D.view(-1, X_3D.size(-1))
        X_3D_sz = X_3D.size()
        X_int8, state_X = quantize_rowwise(X)
        del X
        W_int8, state_W = quantize_global(W)
        ctx.save_for_backward = (X_int8, state_X, W_int8, state_W)
        return int8_matmul_mixed_dequantize(X_int8, W_int8.t(), state_X, state_W, bias).view(*X_3D_sz[:-1], -1)

    @staticmethod
    def backward(ctx, G_3D):
        G = G_3D.reshape(-1, G_3D.size(-1))
        G_3D_sz = G_3D.size()
        grad_X = grad_W = grad_bias = None
        X_int8, state_X, W_int8, state_W = ctx.save_for_backward
        if ctx.needs_input_grad[1]:
            real_X = dequantize_rowwise(X_int8, state_X)
            del X_int8
            grad_W = torch.matmul(G.t(), real_X.to(G.dtype))
            del real_X
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)
        if ctx.needs_input_grad[0]:
            G_int8, state_G = quantize_rowwise(G)
            del G
            W_int8 = W_int8.t().contiguous()
            grad_X = int8_matmul_mixed_dequantize(G_int8, W_int8.t(), state_G, state_W, None).view(*G_3D_sz[:-1], -1)
        return (grad_X, grad_W, grad_bias)