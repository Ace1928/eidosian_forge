import torch
from ... import cdiv, heuristics, jit
from ... import language as tl
@heuristics({'EVEN_K': lambda nargs: nargs['K'] % nargs['TILE_K'] == 0})
@jit
def _sdd_kernel(A, B, C, stride_za, stride_ha, stride_ma, stride_ak, stride_zb, stride_hb, stride_bk, stride_nb, stride_zc, stride_hc, stride_mc, stride_nc, K, grid_offset, lut, TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr, BLOCK: tl.constexpr, EVEN_K: tl.constexpr):
    block_id = tl.program_id(0) + grid_offset
    lut += block_id * 3
    off_z = tl.program_id(2)
    off_h = tl.load(lut + 0)
    start_am = tl.load(lut + 1)
    offs_am = start_am * BLOCK + tl.arange(0, TILE_M) % BLOCK
    offs_ak = tl.arange(0, TILE_K)
    a_ptrs = A + off_z * stride_za + off_h * stride_ha + offs_am[:, None] * stride_ma + offs_ak[None, :] * stride_ak
    start_bn = tl.load(lut + 2)
    offs_bn = start_bn * BLOCK + tl.arange(0, TILE_N) % BLOCK
    offs_bk = tl.arange(0, TILE_K)
    b_ptrs = B + off_z * stride_zb + off_h * stride_hb + offs_bn[None, :] * stride_nb + offs_bk[:, None] * stride_bk
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(K, 0, -TILE_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_ak[None, :] < k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_bk[:, None] < k, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk
    c = acc.to(C.dtype.element_ty)
    offs_cm = tl.arange(0, TILE_M) % BLOCK
    offs_cn = tl.arange(0, TILE_N) % BLOCK
    pc = C + off_z * stride_zc + block_id * stride_hc + offs_cm[:, None] * stride_mc + offs_cn[None, :] * stride_nc
    tl.store(pc, c, mask=True)