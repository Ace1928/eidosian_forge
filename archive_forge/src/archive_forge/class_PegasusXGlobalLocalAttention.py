import dataclasses
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_pegasus_x import PegasusXConfig
class PegasusXGlobalLocalAttention(nn.Module):
    """Global + Local attention. For use with Encoder only."""

    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float=0.0, is_decoder: bool=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim ** (-0.5)
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, token_hidden_states: torch.Tensor, global_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        dim = DimensionInfo(batch_size=token_hidden_states.shape[0], seq_len=token_hidden_states.shape[1], block_size=self.block_size, num_heads=self.num_heads, hidden_dim=token_hidden_states.shape[2], dim_per_head=self.head_dim, num_blocks=token_hidden_states.shape[1] // self.block_size, global_len=global_hidden_states.shape[1], padded_seq_len=token_hidden_states.shape[1])
        local_q = self._shape(self.q_proj(token_hidden_states) * self.scaling, seq_len=dim.padded_seq_len, bsz=dim.batch_size)
        local_k = self._shape(self.k_proj(token_hidden_states), seq_len=dim.padded_seq_len, bsz=dim.batch_size)
        local_v = self._shape(self.v_proj(token_hidden_states), seq_len=dim.padded_seq_len, bsz=dim.batch_size)
        global_q = self._shape(self.q_proj(global_hidden_states) * self.scaling, seq_len=dim.global_len, bsz=dim.batch_size)
        global_k = self._shape(self.k_proj(global_hidden_states), seq_len=dim.global_len, bsz=dim.batch_size)
        global_v = self._shape(self.v_proj(global_hidden_states), seq_len=dim.global_len, bsz=dim.batch_size)
        global_attn_output, global_attn_probs = self.compute_global_attention_representations(global_q=global_q, global_k=global_k, global_v=global_v, local_k=local_k, local_v=local_v, mask=attention_mask, dim=dim)
        local_attn_output, local_attn_probs = self.compute_local_attention_representations(global_k=global_k, global_v=global_v, local_q=local_q, local_k=local_k, local_v=local_v, mask=attention_mask, dim=dim)
        global_attn_output = global_attn_output.transpose(1, 2).contiguous().view(dim.batch_size, dim.global_len, dim.hidden_dim)
        global_attn_output = self.out_proj(global_attn_output)
        local_attn_output = local_attn_output.permute(0, 2, 3, 1, 4).contiguous()
        local_attn_output = local_attn_output.view(dim.batch_size, dim.padded_seq_len, dim.hidden_dim)
        local_attn_output = self.out_proj(local_attn_output)
        if output_attentions:
            attn_probs = {'global': global_attn_probs, 'local': local_attn_probs}
        else:
            attn_probs = None
        return (local_attn_output, global_attn_output, attn_probs)

    def compute_global_attention_representations(self, global_q, global_k, global_v, local_k, local_v, mask, dim: DimensionInfo):
        """Compute attention representations for global tokens.

        Global tokens will attend to both global tokens as well as all input sequence tokens. Because the input
        sequence tokens are arranged in blocks for local attention, we unblock them and compute attention.

        Args:
            global_q (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                query vectors from global tokens
            global_k (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                key vectors from global tokens
            global_v (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                value vectors from global tokens
            local_k (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                key vectors from local tokens
            local_v (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                value vectors from local tokens
            mask (`torch.FloatTensor`) of shape [batch_size, padded_seq_len]: attention mask
            dim (DimensionInfo): DimensionInfo wrapper for dimensions

        Returns:
            output of shape `[batch_sizes, length, features]`. where length will be padded to a multiple of block_size
        """
        global_and_local_k = torch.cat([global_k, local_k], dim=2)
        global_and_local_v = torch.cat([global_v, local_v], dim=2)
        extended_mask = nn.functional.pad(mask, pad=(dim.global_len, 0), value=0)
        attn_weights = torch.einsum('BHGF,BHXF->BHGX', global_q, global_and_local_k)
        attn_weights = attn_weights + extended_mask[:, None, None, :]
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        attn_output = torch.einsum('BHGX,BHXF->BHGF', attn_probs, global_and_local_v)
        return (attn_output, attn_probs)

    def compute_local_attention_representations(self, global_k, global_v, local_q, local_k, local_v, mask, dim: DimensionInfo):
        """Compute attention representations for local tokens.

        Local tokens will attend to both global tokens as well as all other tokens within the same local block. Hence,
        we need to tile and concatenate the global tokens to every local block

        Args:
            global_k (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                key vectors from global tokens
            global_v (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                value vectors from global tokens
            local_q (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                query vectors from local tokens
            local_k (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                key vectors from local tokens
            local_v (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                value vectors from local tokens
            mask (`torch.FloatTensor`) of shape [batch_size, padded_seq_len]: attention mask
            dim (DimensionInfo): DimensionInfo wrapper for dimensions

        Returns:
            output of shape `[batch_sizes, length, features]`. where length will be padded to a multiple of block_size
        """
        blocked_local_q = local_q.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
        blocked_local_k = local_k.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
        blocked_local_v = local_v.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
        extended_mask = nn.functional.pad(mask.view(dim.batch_size, dim.num_blocks, dim.block_size), pad=(dim.global_len, 0), value=0)
        blocked_local2global = torch.einsum('BHNKF,BHGF->BHNKG', blocked_local_q, global_k)
        blocked_local2local = torch.einsum('BHNKF,BHNXF->BHNKX', blocked_local_q, blocked_local_k)
        attn_weights = torch.cat([blocked_local2global, blocked_local2local], dim=-1)
        attn_weights = attn_weights + extended_mask[:, None, :, None, :]
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        local2global_attn_probs = attn_probs[:, :, :, :, :dim.global_len]
        local2local_attn_probs = attn_probs[:, :, :, :, dim.global_len:]
        local2global_attn_output = torch.einsum('BHNKG,BHGF->BHNKF', local2global_attn_probs, global_v)
        local2local_attn_output = torch.einsum('BHNKX,BHNXF->BHNKF', local2local_attn_probs, blocked_local_v)
        attn_output = local2global_attn_output + local2local_attn_output
        return (attn_output, attn_probs)