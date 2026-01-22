import itertools
from typing import List, Tuple
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
def _launch_triton_matmul(a: List[List[torch.Tensor]], b: List[List[torch.Tensor]], c: List[List[torch.Tensor]], ms: List[int], ns: List[int], ks: List[int]) -> None:
    strides_am, strides_ak = _get_strides(a, 'first operand', 'm', 'k')
    strides_bk, strides_bn = _get_strides(b, 'second operand', 'k', 'n')
    strides_cm, strides_cn = _get_strides(c, 'output', 'm', 'n')
    ACC_TYPE = tl.float32 if c[0][0].dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32

    def grid(META):
        return (sum((triton.cdiv(m, META['BLOCK_M']) for m in ms)) * sum((triton.cdiv(n, META['BLOCK_N']) for n in ns)), META['SPLIT_K'])
    _xformers_tiled_matmul_kernel[grid](*[a[min(i, len(a) - 1)][min(j, len(a[0]) - 1)] for i in range(3) for j in range(3)], *[b[min(i, len(b) - 1)][min(j, len(b[0]) - 1)] for i in range(3) for j in range(3)], *[c[min(i, len(c) - 1)][min(j, len(c[0]) - 1)] for i in range(3) for j in range(3)], *[ms[i] if len(ms) > i else 0 for i in range(3)], *[ns[i] if len(ns) > i else 0 for i in range(3)], *[ks[i] if len(ks) > i else 0 for i in range(3)], *strides_am, *strides_ak, *strides_bk, *strides_bn, *strides_cm, *strides_cn, ACC_TYPE=ACC_TYPE)