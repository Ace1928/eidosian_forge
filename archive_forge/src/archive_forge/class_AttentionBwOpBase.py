import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
class AttentionBwOpBase(AttentionOpBase):
    ERROR_ATOL: Mapping[torch.dtype, float] = {torch.float: 0.0009, torch.half: 0.2, torch.bfloat16: 0.9}
    ERROR_RTOL: Mapping[torch.dtype, float] = {torch.float: 0.0001, torch.half: 0.02, torch.bfloat16: 0.1}
    SUPPORTS_ATTN_BIAS_GRAD = False

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(AttentionBwOpBase, cls).not_supported_reasons(d)
        if isinstance(d.attn_bias, torch.Tensor) and d.attn_bias.requires_grad and (not cls.SUPPORTS_ATTN_BIAS_GRAD):
            reasons.append('Computing the bias gradient is not supported (attn_bias.requires_grad = True)')
        return reasons

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        raise NotImplementedError()

    @classmethod
    def attn_operator_flop(cls, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool=False, seqstart_k: Optional[torch.Tensor]=None, seqstart_q: Optional[torch.Tensor]=None) -> int:
        """
        Computes total flops for the attention
        Assumes inputs in format BMHK
        """
        assert query.ndim == 4
        if seqstart_q is not None:
            seqstart_q_py = seqstart_q.tolist()
        else:
            seqstart_q_py = [0, query.shape[1]]
        if seqstart_k is not None:
            seqstart_k_py = seqstart_k.tolist()
        else:
            seqstart_k_py = [0, key.shape[1]]
        total_flop = 0
        for q_start, q_end, k_start, k_end in zip(seqstart_q_py, seqstart_q_py[1:], seqstart_k_py, seqstart_k_py[1:]):
            num_q = q_end - q_start
            num_kv = k_end - k_start
            Kqk = query.shape[-1]
            Kv = value.shape[-1]
            total_flop += num_q * num_kv * Kqk * 2
            total_flop += num_kv * num_q * Kv * 2
            total_flop += num_q * Kv * num_kv * 2
            total_flop += num_q * Kqk * num_kv * 2
            total_flop += num_q * Kqk * num_kv * 2
        total_flop = total_flop * value.shape[2] * value.shape[0]
        if causal:
            total_flop //= 2
        return total_flop