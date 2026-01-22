import torch
from .. import cdiv, jit
from .. import language as tl
class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, sequence_parallel=False):
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError('Flash attention currently only supported for compute capability >= 80')
        BLOCK_M = 128
        BLOCK_N = 64
        Lq, Lk, Lv = (q.shape[-1], k.shape[-1], v.shape[-1])
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](q, k, v, sm_scale, L, o, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), o.stride(0), o.stride(1), o.stride(2), o.stride(3), q.shape[0], q.shape[1], q.shape[2], q.shape[0] * q.shape[1] * q.shape[2], BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk, IS_CAUSAL=causal, num_warps=num_warps, num_stages=4)
        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.sequence_parallel = sequence_parallel
        return o

    @staticmethod
    def backward(ctx, do):
        capability = torch.cuda.get_device_capability()
        MMA_V3 = capability[0] >= 9
        BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        sequence_parallel = ctx.sequence_parallel
        seq_len_kv = k.shape[2]
        do = do.contiguous()
        if sequence_parallel:
            replicas = cdiv(seq_len_kv, BLOCK)
            new_dq_shape = (replicas,) + q.shape
            dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
        else:
            dq = torch.zeros_like(q, dtype=q.dtype)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        _bwd_preprocess[cdiv(q.shape[2], BLOCK) * ctx.grid[1],](o, do, delta, BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL)
        _bwd_kernel[ctx.grid[1], cdiv(seq_len_kv, BLOCK) if sequence_parallel else 1](q, k, v, ctx.sm_scale, o, do, dq, dk, dv, L, delta, o.numel(), q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), q.shape[0], q.shape[1], q.shape[2], q.shape[0] * q.shape[1] * q.shape[2], cdiv(seq_len_kv, BLOCK) * q.shape[0] * q.shape[1] * q.shape[2], BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=ctx.BLOCK_DMODEL, SEQUENCE_PARALLEL=sequence_parallel, CAUSAL=ctx.causal, MMA_V3=MMA_V3, num_warps=8, num_stages=1)
        if len(dq.shape) == 5:
            dq = dq.sum(dim=0)
        return (dq, dk, dv, None, None, None)