import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
from ....utils import logging
from .configuration_mctct import MCTCTConfig
class MCTCTSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_dim
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.max_position_embeddings = config.max_position_embeddings
        self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def reshape_fortran(self, x, shape):
        if len(x.shape) > 0:
            x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

    def relative_position_embedding_rotate(self, scores):
        scores = scores.permute(0, 2, 3, 1)
        batch, hidden_state, seq_len, heads = scores.shape
        scores = torch.cat((scores, torch.zeros((batch, seq_len, seq_len, heads), device=scores.device)), dim=1)
        scores = self.reshape_fortran(scores, [batch, (hidden_state + seq_len) * seq_len, 1, heads])
        scores = scores[:, :(seq_len + hidden_state - 1) * seq_len]
        scores = self.reshape_fortran(scores, [batch, hidden_state + seq_len - 1, seq_len, heads])
        halfpoint = hidden_state // 2
        scores = scores[:, halfpoint:halfpoint + seq_len].transpose(1, 2)
        return scores.permute(0, 3, 1, 2)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_query_layer = mixed_query_layer / math.sqrt(self.attention_head_size)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        positional_embedding = self.distance_embedding.weight
        relative_position_scores = torch.einsum('lh, bche -> bcle', positional_embedding, query_layer.transpose(2, 3))
        relative_position_scores = self.relative_position_embedding_rotate(relative_position_scores)
        attention_scores = attention_scores + relative_position_scores
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).flatten(start_dim=-2)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs