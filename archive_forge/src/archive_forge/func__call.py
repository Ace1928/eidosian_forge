import torch
from .. import Config, autotune, cdiv, heuristics, jit
from .. import language as tl
from .matmul_perf_model import early_config_prune, estimate_matmul_time
@staticmethod
def _call(a, b, dot_out_dtype, allow_tf32, fp8_fast_accum):
    device = a.device
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], 'incompatible dimensions'
    M, K = a.shape
    _, N = b.shape
    if a.dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5] or b.dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
        c_dtype = torch.float16
    elif a.dtype in [torch.int8] or b.dtype in [torch.int8]:
        c_dtype = torch.int32
    else:
        c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    if dot_out_dtype is None:
        if c_dtype in [torch.float16, torch.float32, torch.bfloat16]:
            dot_out_dtype = tl.float32
        else:
            dot_out_dtype = tl.int32
    else:
        assert isinstance(dot_out_dtype, torch.dtype), 'dot_out_dtype must be a torch.dtype'
        if dot_out_dtype == torch.float16:
            dot_out_dtype = tl.float16
        elif dot_out_dtype in [torch.float32, torch.bfloat16]:
            dot_out_dtype = tl.float32
        else:
            dot_out_dtype = tl.int32
    ab_dtype = True
    if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [tl.float8e4nv, tl.float8e5]:
        ab_dtype = False
    if a.dtype in [torch.int8] and b.dtype in [torch.int8]:
        ab_dtype = False
    grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    _kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), dot_out_dtype=dot_out_dtype, allow_tf32=allow_tf32, fp8_fast_accum=fp8_fast_accum, GROUP_M=8, AB_DTYPE=ab_dtype)
    return c