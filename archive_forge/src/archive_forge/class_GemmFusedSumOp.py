from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind
@register_operator
class GemmFusedSumOp(BaseOperator):
    OPERATOR = get_xformers_operator('gemm_fused_operand_sum')
    OPERATOR_CATEGORY = 'swiglu'
    NAME = 'gemm_fused_operand_sum'

    @classmethod
    def operator_flop(cls, a: torch.Tensor, b: torch.Tensor, out1, out2) -> int:
        M, N, K = (a.shape[0], b.shape[1], a.shape[1])
        return M * N * K * 2