import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
def _flash_bwd(grad, query, key, value, out, lse, dq, dk, dv, cu_seq_lens_q, cu_seq_lens_k, max_seq_len_q, max_seq_len_k, p, softmax_scale, is_causal, window_left, window_right, rng_state):
    if cu_seq_lens_k is None:
        assert cu_seq_lens_q is None
        _C_flashattention.bwd(grad, query, key, value, out, lse, dq, dk, dv, None, p, softmax_scale, is_causal, window_left, window_right, False, None, rng_state)
    else:
        _C_flashattention.varlen_bwd(grad, query, key, value, out, lse, dq, dk, dv, cu_seq_lens_q, cu_seq_lens_k, None, max_seq_len_q, max_seq_len_k, p, softmax_scale, False, is_causal, window_left, window_right, False, None, rng_state)
    return (dq, dk, dv)