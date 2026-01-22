import torch
from ... import cdiv, heuristics, jit
from ... import language as tl
@jit
def _dsd_kernel(A, B, C, stride_az, stride_ha, stride_am, stride_ak, stride_zb, stride_hb, stride_bk, stride_bn, stride_zc, stride_hc, stride_cm, stride_cn, DS0, DS1, lut, TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE_M)
    pidz = tl.program_id(2)
    header = lut + pid_n * 4
    offset = tl.load(header + 0)
    K = tl.load(header + 1)
    column = tl.load(header + 2)
    off_h = tl.load(header + 3)
    pinc = lut + offset
    block_id = tl.load(pinc + 1)
    block_id = tl.multiple_of(block_id, 8)
    offs_am = tl.arange(0, TILE_M)
    offs_ak = tl.arange(0, TILE_K)
    pa = A + pidz * stride_az + block_id * stride_ha + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_bn = pid_m * TILE_N + tl.arange(0, TILE_N)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn % DS0, TILE_N), TILE_N)
    start_bk = tl.load(pinc)
    start_bk = tl.multiple_of(start_bk, 8)
    offs_bk = start_bk + tl.arange(0, TILE_K)
    pb = B + pidz * stride_zb + off_h * stride_hb + offs_bn[None, :] * stride_bn + offs_bk[:, None] * stride_bk
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    pinc += 2
    inc_a = tl.load(pinc + 1)
    inc_a = tl.multiple_of(inc_a, 8)
    inc_b = tl.load(pinc)
    inc_b = tl.multiple_of(inc_b, 8)
    for k in range(K, 0, -TILE_K):
        a = tl.load(pa)
        b = tl.load(pb)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        pa += inc_a
        pb += inc_b * stride_bk
        pinc += 2
        inc_a = tl.load(pinc + 1)
        inc_a = tl.multiple_of(inc_a, 8)
        inc_b = tl.load(pinc)
        inc_b = tl.multiple_of(inc_b, 8)
    c = acc.to(C.dtype.element_ty)
    offs_cm = column * TILE_M + tl.arange(0, TILE_M)
    offs_cn = pid_m * TILE_N + tl.arange(0, TILE_N)
    pc = C + off_h * stride_hc + pidz * stride_zc + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(pc, c, mask=offs_cn[None, :] < DS0)