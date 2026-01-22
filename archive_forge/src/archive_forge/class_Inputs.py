import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
@dataclass
class Inputs:
    """
    Stores inputs to the `memory_efficient_attention` operators
    """
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]] = None
    p: float = 0.0
    scale: Optional[float] = None
    output_dtype: Optional[torch.dtype] = None
    is_partial: bool = False

    @property
    def device(self) -> torch.device:
        return self.query.device

    @property
    def scale_float(self) -> float:
        return self.query.shape[-1] ** (-0.5) if self.scale is None else self.scale

    def get_qkv_in_bmghk(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.query.ndim == 5:
            return (self.query, self.key, self.value)
        if self.query.ndim == 4:
            return (self.query.unsqueeze(2), self.key.unsqueeze(2), self.value.unsqueeze(2))
        if self.value.ndim == 3:
            return (self.query[:, :, None, None], self.key[:, :, None, None], self.value[:, :, None, None])
        assert False

    def normalize_bmhk(self) -> Tuple[int, ...]:
        if self.query.ndim not in [3, 4, 5]:
            raise ValueError(f'Invalid shape for query: {self.query.shape}. Expected shape [batch, seqlen, head_groups, num_heads_per_group, K], [batch, seqlen, num_heads, K], or [batch, seqlen, K].')
        if self.value.dtype == torch.int32:
            output_shape = tuple(self.query.shape)
        else:
            output_shape = self.query.shape[:-1] + (self.value.shape[-1],)
        if self.query.ndim == 3:
            self.query = self.query.unsqueeze(2)
            self.key = self.key.unsqueeze(2)
            self.value = self.value.unsqueeze(2)
            self.attn_bias = _attn_bias_apply(self.attn_bias, partial(torch.unsqueeze, dim=1))
        return output_shape

    def validate_inputs(self) -> None:
        qkv = (self.query, self.key, self.value)
        if self.query.ndim not in (3, 4, 5) or any((x.ndim != self.query.ndim for x in qkv)):
            raise ValueError(f'Query/Key/Value should all have BMGHK, BMHK or BMK shape.\n  query.shape: {self.query.shape}\n  key.shape  : {self.key.shape}\n  value.shape: {self.value.shape}')
        if any((x.device != self.query.device for x in qkv)):
            raise ValueError('Query/Key/Value should all be on the same device')
        quantized_dtypes = self.key.dtype == self.value.dtype == torch.int32
        non_quantized_dtypes = all((x.dtype == self.query.dtype for x in qkv))
        if not (quantized_dtypes or non_quantized_dtypes):
            raise ValueError(f'Query/Key/Value should either all have the same dtype, or (in the quantized case) Key/Value should have dtype torch.int32\n  query.dtype: {self.query.dtype}\n  key.dtype  : {self.key.dtype}\n  value.dtype: {self.value.dtype}')
        if self.query.ndim == 3 and (not _is_bias_type_supported_in_BMK(type(self.attn_bias))):
            raise ValueError(f'Please provide inputs in BMHK format rather than BMK when using bias type `{type(self.attn_bias).__name__}`')
        attn_bias_t: Optional[torch.Tensor] = None
        if isinstance(self.attn_bias, torch.Tensor):
            attn_bias_t = self.attn_bias
        if isinstance(self.attn_bias, LowerTriangularMaskWithTensorBias):
            attn_bias_t = self.attn_bias._bias
        if self.query.ndim == 4 and attn_bias_t is not None:
            expected_shape = (self.query.shape[0], self.query.shape[2], self.query.shape[1], self.key.shape[1])
            if attn_bias_t.shape != expected_shape:
                raise ValueError(f'Invalid shape for attention bias: {attn_bias_t.shape} (expected {expected_shape})\n  query.shape: {self.query.shape}\n  key.shape  : {self.key.shape}\n  value.shape: {self.value.shape}')
        if isinstance(self.attn_bias, BlockDiagonalMask):
            if any((x.shape[0] != 1 for x in qkv)):
                raise ValueError(f'Expected batch_size=1 when using block-diagonal bias\n  query.shape: {self.query.shape}\n  key.shape  : {self.key.shape}\n  value.shape: {self.value.shape}')
        if self.p < 0.0 or self.p > 1.0:
            raise ValueError(f'Invalid dropout probability: p={self.p}')
        B, Mq = self.query.shape[:2]
        K = self.query.shape[-1]
        B, Mkv = self.key.shape[:2]
        Kv = self.value.shape[-1]
        quantized_kv_cache = self.value.dtype == torch.int32
        key_embed_dim = Kv if quantized_kv_cache else K
        valid_shapes = True
        if self.query.ndim == 3:
            valid_shapes = self.query.shape == (B, Mq, K) and self.key.shape == (B, Mkv, K) and (self.value.shape == (B, Mkv, Kv))
        H = self.query.shape[-2]
        if self.query.ndim == 4:
            valid_shapes = self.query.shape == (B, Mq, H, K) and self.key.shape == (B, Mkv, H, key_embed_dim) and (self.value.shape == (B, Mkv, H, Kv))
        G = self.query.shape[2]
        if self.query.ndim == 5:
            valid_shapes = self.query.shape == (B, Mq, G, H, K) and self.key.shape == (B, Mkv, G, H, key_embed_dim) and (self.value.shape == (B, Mkv, G, H, Kv))
        if not valid_shapes:
            raise ValueError(f"Incompatible shapes for attention inputs:\n  query.shape: {self.query.shape}\n  key.shape  : {self.key.shape}\n  value.shape: {self.value.shape}\nHINT: We don't support broadcasting, please use `expand` yourself before calling `memory_efficient_attention` if you need to")

    def get_output_dtype(self) -> torch.dtype:
        if self.output_dtype is None:
            if self.is_partial and self.query.dtype is not torch.float64:
                return torch.float32
            return self.query.dtype
        return self.output_dtype