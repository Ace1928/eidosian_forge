import math
import random
from typing import List, Optional, Sequence, Tuple, Type
import torch
from xformers.ops import fmha
from xformers.ops.fmha.common import AttentionOpBase
def _rand_seqlens(r: random.Random, bs: int, q_len: int, kv_len: int, max_q_minus_k: Optional[int]) -> Tuple[Sequence[int], Sequence[int]]:
    """
    Generates lists of lengths of query blocks and corresponding key blocks.
    The total number of queries will be bs * q_len and the
    total number of keys will be bs * kv_len.
    max_q_minus_k: maximum allowed num_queries - num_keys.
        For "bottom-right" masks it's 0, we need to have more keys than
        queries, otherwise some queries have no keys to attend to.
        For BlockDiagonalCausalMask it's None, there is no constraint
        on num_queries - num_keys.
        For BlockDiagonalCausalLocalAttentionMask it's equal
        to the window size.
    """
    if max_q_minus_k == 0:
        assert kv_len >= q_len
    q_len *= bs
    kv_len *= bs
    seqlens_q: List[int] = []
    seqlens_k: List[int] = []
    step_q = [max(1, q_len // 10), max(2, q_len // 2)]
    step_k = [max(1, kv_len // 10), max(2, kv_len // 2)]
    while sum(seqlens_q) < q_len and sum(seqlens_k) < kv_len:
        if max_q_minus_k is None:
            num_queries = r.randrange(*step_q)
            seqlens_q.append(num_queries)
            seqlens_k.append(r.randrange(*step_k))
        else:
            keys_left = kv_len - sum(seqlens_k, 0)
            queries_left = q_len - sum(seqlens_q, 0)
            assert keys_left >= queries_left - max_q_minus_k, f'keys_left={keys_left!r} queries_left={queries_left!r} max_q_minus_k={max_q_minus_k!r} kv_len={kv_len!r} q_len={q_len!r} seqlens_k={seqlens_k!r} seqlens_q={seqlens_q!r}'
            max_queries_to_take = min(queries_left, keys_left + max_q_minus_k)
            num_queries = r.randrange(1, max_queries_to_take + 1)
            seqlens_q.append(num_queries)
            extra_keys_available = keys_left - queries_left + max_q_minus_k + 1
            assert extra_keys_available >= 0
            if extra_keys_available > 0:
                seqlens_k.append(num_queries + r.randrange(0, extra_keys_available))
            else:
                seqlens_k.append(num_queries)
    seqlens_q[-1] = q_len - sum(seqlens_q[:-1])
    seqlens_k[-1] = kv_len - sum(seqlens_k[:-1])
    return (seqlens_q, seqlens_k)