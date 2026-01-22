from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
def _check_large_shapes(reasons: List[str], inp: Inputs) -> None:
    """CK kernel throws "Memory access fault by GPU node-2" when B * T >= 2**20, might be some index overflow.
    To reproduce, remove this function and run benchmark_mem_eff_attention with ParlAI model shape (256, 4096, 16, 64).
    This needs further debugging, for now let's not support such shapes.
    """
    b_t_limit = 1024 ** 2
    q_too_large = inp.query.shape[0] * inp.query.shape[1] >= b_t_limit
    k_too_large = inp.key.shape[0] * inp.key.shape[1] >= b_t_limit
    v_too_large = inp.value.shape[0] * inp.value.shape[1] >= b_t_limit
    if q_too_large or k_too_large or v_too_large:
        reasons.append('Input is too large: product of first two dimensions of q/k/v must be < 2**20')