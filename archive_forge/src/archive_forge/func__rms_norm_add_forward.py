import torch
import triton
import triton.language as tl
def _rms_norm_add_forward(x, y, attn_norm_weights, eps):
    if not x.is_contiguous():
        raise ValueError('x must be contiguous')
    if not y.is_contiguous():
        raise ValueError('y must be contiguous')
    if attn_norm_weights is not None:
        if not attn_norm_weights.is_contiguous():
            raise ValueError('weights must be contiguous')
    out = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    y_arg = y.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    with torch.cuda.device(x.device):
        _rms_norm_add_kernel[M,](x_arg, y_arg, out, attn_norm_weights, eps, x_arg.stride(0), N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, INCLUDE_WEIGHT=attn_norm_weights is not None)
    return out