import itertools
from typing import List, Optional, Set, Tuple, cast
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
@triton.autotune(configs=TRITON_CONFIGS, key=['M', 'N1', 'N2', 'N3', 'K'], prune_configs_by={'early_config_prune': early_config_prune, 'perf_model': our_estimate_matmul_time, 'top_k': 10}, reset_to_zero=['blocks_done_counters'])
@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0})
@triton.jit(do_not_specialize=['wait_counters', 'blocks_done_counters', 'write_counters', 'do_wait', 'do_write', 'direction', 'stripe', 'seq_num', 'num_stripes', '_wait', 'my_rank', 'world_size', 'timeout_ns'], debug=True)
def _xformers_seqpar_matmul_kernel(A_my_shard, A, B1, B2, B3, C1, C2, C3, C1_my_shard, C2_my_shard, C3_my_shard, wait_counters, blocks_done_counters, write_counters, M, N1, N2, N3, K, stride_am, stride_ak, stride_bk1, stride_bk2, stride_bk3, stride_bn1, stride_bn2, stride_bn3, stride_cm1, stride_cm2, stride_cm3, stride_cn1, stride_cn2, stride_cn3, do_wait, do_write, direction, stripe, seq_num, num_stripes, _wait, my_rank, world_size, timeout_ns, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, ACC_TYPE: tl.constexpr):
    A, B, C, M, N, stride_bk, stride_bn, stride_cm, stride_cn, pid_m, pid_n, other_rank, num_blocks_2d = determine_tile(A, B1, B2, B3, C1, C2, C3, A_my_shard, C1_my_shard, C2_my_shard, C3_my_shard, M, N1, N2, N3, my_rank, world_size, direction, stride_am, stride_bk1, stride_bk2, stride_bk3, stride_bn1, stride_bn2, stride_bn3, stride_cm1, stride_cm2, stride_cm3, stride_cn1, stride_cn2, stride_cn3, BLOCK_M, BLOCK_N, GROUP_M)
    pid_z = tl.program_id(1)
    wait_for_recv(seq_num, wait_counters, other_rank, my_rank, stripe, num_stripes, _wait, do_wait, timeout_ns)
    do_matmul(A, B, C, pid_m, pid_n, pid_z, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, SPLIT_K, EVEN_K)
    trigger_send(seq_num, blocks_done_counters, write_counters, other_rank, my_rank, num_stripes, stripe, num_blocks_2d * SPLIT_K, _wait, do_write)