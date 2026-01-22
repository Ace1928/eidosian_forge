import logging
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import (
from xformers.components.attention.utils import (
@register_attention('nystrom', NystromSelfAttentionConfig)
class NystromAttention(Attention):

    def __init__(self, dropout: float, num_heads: int, num_landmarks: int=64, landmark_pooling: Optional[nn.Module]=None, causal: bool=False, use_razavi_pinverse: bool=True, pinverse_original_init: bool=False, inv_iterations: int=6, v_skip_connection: Optional[nn.Module]=None, conv_kernel_size: Optional[int]=None, *args, **kwargs):
        """
        Nystrom attention mechanism, from Nystromformer_.
        ::

            "A Nystrom-based Algorithm for Approximating Self-Attention."
            Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., Singh, V. (2021)

            Reference codebase: https://github.com/mlpen/Nystromformer

        .. _Nystromformer: https://arxiv.org/pdf/2102.03902.pdf

        """
        super().__init__()
        self.requires_separate_masks = True
        self.num_landmarks = num_landmarks
        self.num_heads = num_heads
        self.use_razavi_pinverse = use_razavi_pinverse
        self.pinverse_original_init = pinverse_original_init
        self.inv_iterations = inv_iterations
        self.attn_drop = nn.Dropout(dropout)
        self.skip_connection = v_skip_connection
        self.causal = causal
        if self.skip_connection is None and conv_kernel_size is not None:
            self.skip_connection = nn.Conv2d(in_channels=self.num_heads, out_channels=self.num_heads, kernel_size=(conv_kernel_size, 1), padding=(conv_kernel_size // 2, 0), bias=False, groups=self.num_heads)
        if landmark_pooling is not None:
            self.landmark_pooling = landmark_pooling
        else:
            self.landmark_pooling = AvgPool(n=self.num_landmarks)
        self.causal_mask_1: Optional[torch.Tensor] = None
        self.causal_mask_2: Optional[torch.Tensor] = None
        self.causal_mask_3: Optional[torch.Tensor] = None
        self.supports_attention_mask = False
        self.supports_key_padding_mask = True

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None, *args, **kwargs):
        """
        key_padding_mask    Only a key padding mask is accepted here. The size must be (batch size, sequence length) or
                            (batch size * num_heads, 1, sequence length). If dimensions are not correct, the mask will
                            be ignored. An additive mask is expected, meaning float values using "-inf" to mask values
        """
        batched_dim = k.size(0)
        seq_len = k.size(-2)
        tt = {'dtype': q.dtype, 'device': q.device}
        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                logger.warning('Bool mask found, but an additive mask is expected. Converting but this is slow')
                key_padding_mask = bool_mask_to_additive(key_padding_mask)
            if key_padding_mask.ndim == 2:
                key_padding_mask = reshape_key_padding_mask(key_padding_mask, batched_dim)
            zeros = torch.zeros_like(key_padding_mask)
            ones = torch.ones_like(key_padding_mask)
            is_masked = torch.isinf(-key_padding_mask)
            _mask = torch.where(is_masked, zeros, ones)
            _mask = _mask.transpose(2, 1)
            assert _mask.shape == (batched_dim, q.shape[1], 1)
            q = q * _mask
            k = k * _mask
            assert key_padding_mask.size() == (batched_dim, 1, seq_len), f'key_padding_mask has invalid dimensions {key_padding_mask.size()}. Must have dimensions {(batched_dim, 1, seq_len)} or (batch_size, {seq_len}).'
        if self.num_landmarks >= seq_len:
            mask: Optional[torch.Tensor] = None
            if self.causal:
                mask = self._triu_mask(batched_dim, seq_len, seq_len, **tt)
            if key_padding_mask is not None:
                mask = key_padding_mask if mask is None else mask + key_padding_mask
            x = scaled_dot_product_attention(q=q, k=k, v=v, att_mask=mask)
        else:
            q_landmarks = self.landmark_pooling(q)
            k_landmarks = self.landmark_pooling(k)
            if self.causal and (self.causal_mask_1 is None or (batched_dim, seq_len, self.num_landmarks) != self.causal_mask_1.size()):
                self.causal_mask_1 = self._triu_mask(batched_dim, seq_len, self.num_landmarks, **tt)
                self.causal_mask_2 = self._triu_mask(batched_dim, self.num_landmarks, self.num_landmarks, **tt)
                self.causal_mask_3 = self._triu_mask(batched_dim, self.num_landmarks, seq_len, **tt)
            mask_3: Optional[torch.Tensor] = self.causal_mask_3
            if key_padding_mask is not None:
                mask_3 = key_padding_mask if mask_3 is None else mask_3 + key_padding_mask
            kernel_1 = scaled_query_key_softmax(q=q, k=k_landmarks, att_mask=None)
            kernel_2 = scaled_query_key_softmax(q=q_landmarks, k=k_landmarks, att_mask=None)
            kernel_3 = scaled_dot_product_attention(q=q_landmarks, k=k, v=v, att_mask=mask_3)
            kernel_2_inv = iterative_pinv(kernel_2, self.inv_iterations, self.pinverse_original_init) if self.use_razavi_pinverse else torch.linalg.pinv(kernel_2)
            x = torch.matmul(torch.matmul(kernel_1, kernel_2_inv), kernel_3)
        if self.skip_connection:
            v_conv = self.skip_connection(v.reshape(-1, self.num_heads, v.size(-2), v.size(-1)))
            x += v_conv.reshape(-1, v_conv.size(-2), v_conv.size(-1))
        x = self.attn_drop(x)
        return x

    def _triu_mask(self, dim_1: int, dim_2: int, dim_3: int, **kwargs) -> torch.Tensor:
        device = kwargs['device']
        dtype = kwargs['dtype']
        return torch.triu(torch.ones(dim_2, dim_3, dtype=dtype, device=device) * float('-inf'), diagonal=1).expand(dim_1, -1, -1)