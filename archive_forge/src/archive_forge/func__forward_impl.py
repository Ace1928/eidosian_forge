import torch
import torch.jit  # this is needed to avoid a circular import
from torch import nn
import torch.nn.functional as nnF
from torch import Tensor
from typing import Optional, Tuple
import warnings
def _forward_impl(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor]=None, need_weights: bool=True, attn_mask: Optional[Tensor]=None, average_attn_weights: bool=True, is_causal: bool=False) -> Tuple[Tensor, Optional[Tensor]]:
    static_k = None
    static_v = None
    if attn_mask is not None and is_causal:
        raise AssertionError('Only allow causal mask or attn_mask')
    if is_causal:
        raise AssertionError('causal mask not supported by AO MHA module')
    if self.batch_first:
        query, key, value = (x.transpose(0, 1) for x in (query, key, value))
    tgt_len, bsz, embed_dim_to_check = query.size()
    assert self.embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
    head_dim = self.embed_dim // self.num_heads
    assert head_dim * self.num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** (-0.5)
    q = self.linear_Q(query)
    k = self.linear_K(key)
    v = self.linear_V(value)
    q = self.q_scaling_product.mul_scalar(q, scaling)
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn('Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
            attn_mask = attn_mask.to(torch.bool)
        assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, f'Only float and bool types are supported for attn_mask, not {attn_mask.dtype}'
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn('Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.')
        key_padding_mask = key_padding_mask.to(torch.bool)
    if self.bias_k is not None and self.bias_v is not None:
        if static_k is None and static_v is None:
            bias_k = self.bias_k
            assert bias_k is not None
            bias_v = self.bias_v
            assert bias_v is not None
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = nnF.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = nnF.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, 'bias cannot be added to static key.'
            assert static_v is None, 'bias cannot be added to static value.'
    else:
        assert self.bias_k is None
        assert self.bias_v is None
    q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * self.num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * self.num_heads
        assert static_v.size(2) == head_dim
        v = static_v
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    if self.add_zero_attn:
        src_len += 1
        k_zeros = torch.zeros((k.size(0), 1) + k.size()[2:])
        if k.is_quantized:
            k_zeros = torch.quantize_per_tensor(k_zeros, k.q_scale(), k.q_zero_point(), k.dtype)
        k = torch.cat([k, k_zeros], dim=1)
        v_zeros = torch.zeros((v.size(0), 1) + k.size()[2:])
        if v.is_quantized:
            v_zeros = torch.quantize_per_tensor(v_zeros, v.q_scale(), v.q_zero_point(), v.dtype)
        v = torch.cat([v, v_zeros], dim=1)
        if attn_mask is not None:
            attn_mask = nnF.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = nnF.pad(key_padding_mask, (0, 1))
    q = self.dequant_q(q)
    k = self.dequant_k(k)
    v = self.dequant_v(v)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
    attn_output_weights = nnF.softmax(attn_output_weights, dim=-1)
    attn_output_weights = nnF.dropout(attn_output_weights, p=self.dropout, training=self.training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
    if self.batch_first:
        attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
    else:
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
    attn_output = self.quant_attn_output(attn_output)
    attn_output = self.out_proj(attn_output)
    attn_output_weights = self.quant_attn_output_weights(attn_output_weights)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)
        return (attn_output, attn_output_weights)
    else:
        return (attn_output, None)