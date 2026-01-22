import torch
from ... import jit
from ... import language as tl
from ... import next_power_of_2
@jit
def _blocksparse_softmax_fwd(Out, A, stride_xz, LUT, R, extent, stride_zr, stride_hr, scale, is_causal, ROW_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr, IS_DENSE: tl.constexpr):
    h = tl.program_id(0)
    m = tl.program_id(1)
    z = tl.program_id(2)
    hm = h * tl.num_programs(1) + m
    lane_n = tl.arange(0, ROW_SIZE) % BLOCK_SIZE
    block_n = tl.arange(0, ROW_SIZE) // BLOCK_SIZE
    header = LUT + hm // BLOCK_SIZE * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    off_a = z * stride_xz
    off_a += (offset + block_n) * BLOCK_SIZE * BLOCK_SIZE
    off_a += m % BLOCK_SIZE * BLOCK_SIZE
    if IS_DENSE:
        ns = tl.arange(0, ROW_SIZE)
    else:
        off_lut = offset + 2 * tl.num_programs(0) * tl.num_programs(1) // BLOCK_SIZE
        start_n = tl.load(LUT + off_lut + block_n, mask=block_n < size, other=0)
        ns = start_n * BLOCK_SIZE + lane_n
    mask = block_n < size
    a = tl.load(A + off_a + lane_n, mask=mask, other=-float('inf'))
    a = a.to(tl.float32)
    out = a
    out *= scale
    if R is not None:
        R += z * stride_zr
        R += h * stride_hr
        off_lo = extent - m - 1 + ns
        mask_lo = (off_lo >= 0) & (off_lo < extent)
        rel_logits = tl.load(R + m * extent + off_lo, mask=mask_lo, other=0.0)
        out += rel_logits
    out = out.to(tl.float32)
    out = tl.where((ns > m) & is_causal, -float('inf'), out)
    out = tl.softmax(out)
    tl.store(Out + off_a + lane_n, out, mask=mask)