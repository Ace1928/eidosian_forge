import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
class XLMProphetNetNgramSelfAttention(nn.Module):

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance
        self.num_attn_heads = config.num_decoder_attention_heads
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        self.head_dim = config.hidden_size // self.num_attn_heads
        self.ngram = config.ngram
        assert self.head_dim * self.num_attn_heads == config.hidden_size, 'config.hidden_size must be divisible by num_attn_heads'
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.relative_pos_embeddings = nn.Linear(config.hidden_size, self.num_buckets * self.num_attn_heads)
        self.onnx_trace = False

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2).contiguous()

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, hidden_states, past_key_value: Optional[Tuple[Tensor]]=None, attention_mask=None, layer_head_mask=None, extended_predict_attention_mask=None, main_relative_position_buckets=None, predict_relative_position_buckets=None, position_ids=None):
        batch_size, ngram_sequence_length, hidden_size = hidden_states.size()
        assert list(hidden_states.size()) == [batch_size, ngram_sequence_length, hidden_size], f'`hidden_states` should be of shape {(batch_size, ngram_sequence_length, hidden_size)}, but is of shape {hidden_states.shape}'
        query_states = self.query_proj(hidden_states)
        key_states = self.key_proj(hidden_states)
        value_states = self.value_proj(hidden_states)
        query_states = query_states / self.head_dim ** 0.5
        query_states = self._shape(query_states, ngram_sequence_length, batch_size)
        key_states = self._shape(key_states, -1, batch_size)
        value_states = self._shape(value_states, -1, batch_size)
        proj_shape = (batch_size, self.num_attn_heads, -1, self.head_dim)
        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        hidden_states_list = hidden_states.chunk(1 + self.ngram, dim=1)
        query_states_list = query_states.chunk(1 + self.ngram, dim=2)
        key_states_list = key_states.chunk(1 + self.ngram, dim=2)
        value_states_list = value_states.chunk(1 + self.ngram, dim=2)
        main_hidden_states, hidden_states_predict_list = (hidden_states_list[0], hidden_states_list[1:])
        main_query_states, predict_query_states_list = (query_states_list[0], query_states_list[1:])
        main_key_states, predict_key_states_list = (key_states_list[0], key_states_list[1:])
        main_value_states, predict_value_states_list = (value_states_list[0], value_states_list[1:])
        if past_key_value is not None:
            prev_main_key_states = past_key_value[0]
            main_key_states = torch.cat((prev_main_key_states, main_key_states), dim=2)
            prev_main_value_states = past_key_value[1]
            main_value_states = torch.cat((prev_main_value_states, main_value_states), dim=2)
        past_key_value = (main_key_states, main_value_states)
        sequence_length = ngram_sequence_length // (1 + self.ngram)
        main_attn_weights = torch.einsum('bntc,bncs->bnts', main_query_states, main_key_states.transpose(2, 3))
        main_relative_pos_embeddings = self.get_main_relative_pos_embeddings(main_hidden_states, main_attn_weights, position_ids, main_relative_position_buckets)
        main_attn_weights = main_attn_weights + main_relative_pos_embeddings
        if attention_mask is not None:
            main_attn_weights = main_attn_weights + attention_mask
        main_attn_probs = softmax(main_attn_weights, dim=-1, onnx_trace=self.onnx_trace).type_as(main_attn_weights)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_attn_heads,), f'Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is {layer_head_mask.size()}'
            main_attn_probs = layer_head_mask.view(1, -1, 1, 1) * main_attn_probs.view(batch_size, self.num_attn_heads, -1, sequence_length)
        main_attn_probs = nn.functional.dropout(main_attn_probs, p=self.attention_dropout, training=self.training)
        main_attn_output = torch.einsum('bntc,bncs->bnts', main_attn_probs, main_value_states)
        main_attn_output = main_attn_output.transpose(1, 2).reshape(batch_size, 1, sequence_length, hidden_size)
        main_attn_output = self.out_proj(main_attn_output)
        predict_query_states = torch.stack(predict_query_states_list, 1).view(batch_size, self.ngram, self.num_attn_heads, sequence_length, self.head_dim)
        predict_key_states = torch.stack([torch.cat([main_key_states, key], 2) for key in predict_key_states_list], 1)
        predict_hidden_states = torch.stack(hidden_states_predict_list, dim=2)
        predict_value_states = torch.cat([torch.cat([main_value_states, v_p], 2).unsqueeze(2) for v_p in predict_value_states_list], 2)
        predict_attn_weights = torch.einsum('bnhtc,bnhsc->bnhts', (predict_query_states, predict_key_states))
        predict_relative_pos_embeddings = self.get_predict_relative_pos_embeddings(predict_hidden_states, predict_attn_weights, position_ids, predict_relative_position_buckets)
        predict_attn_weights = predict_attn_weights + predict_relative_pos_embeddings
        if extended_predict_attention_mask is not None:
            extended_predict_attention_mask = extended_predict_attention_mask.permute(0, 2, 1, 3, 4)
            extended_predict_attention_mask = extended_predict_attention_mask.to(predict_attn_weights.dtype)
            predict_attn_weights = predict_attn_weights + extended_predict_attention_mask
        predict_attn_probs = softmax(predict_attn_weights, dim=-1, onnx_trace=self.onnx_trace).type_as(predict_attn_weights)
        if layer_head_mask is not None:
            assert layer_head_mask.size() == (self.num_attn_heads,), f'Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is {layer_head_mask.size()}'
            predict_attn_probs = layer_head_mask.view(1, 1, -1, 1, 1) * predict_attn_probs
        predict_attn_probs = nn.functional.dropout(predict_attn_probs, p=self.attention_dropout, training=self.training)
        predict_attn_output = torch.einsum('bnhts,bnhsc->bnhtc', (predict_attn_probs, predict_value_states.transpose(1, 2)))
        predict_attn_output = predict_attn_output.transpose(2, 3)
        predict_attn_output = predict_attn_output.reshape(batch_size, self.ngram, sequence_length, hidden_size)
        predict_attn_output = self.out_proj(predict_attn_output)
        attn_output = torch.cat([main_attn_output, predict_attn_output], 1).view(batch_size, -1, hidden_size)
        main_attn_probs = main_attn_probs.view(batch_size, self.num_attn_heads, sequence_length, -1)
        attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)
        return (attn_output, main_attn_probs, predict_attn_probs, past_key_value)

    def get_main_relative_pos_embeddings(self, hidden_states, attn_weights, position_ids, main_relative_position_buckets):
        batch_size, num_attn_heads, tgt_len, src_len = attn_weights.shape
        attn_weights = attn_weights.view(batch_size, num_attn_heads, tgt_len, src_len)
        if main_relative_position_buckets is None:
            batch_size, sequence_length = hidden_states.shape[:2]
            relative_positions = torch.arange(1, attn_weights.shape[-1] + 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, sequence_length, 1).to(position_ids.device)
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            main_relative_position_buckets = compute_relative_buckets(self.num_buckets, self.relative_max_distance, relative_positions, False)
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
        rel_pos_embeddings = rel_pos_embeddings.view(rel_pos_embeddings.shape[:2] + (self.num_buckets, self.num_attn_heads))
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 3, 1, 2)
        rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:3] + (-1,))
        main_relative_position_buckets = main_relative_position_buckets.repeat(1, self.num_attn_heads, 1)
        main_relative_position_buckets = main_relative_position_buckets.view(-1, main_relative_position_buckets.shape[-1])
        main_relative_position_buckets = main_relative_position_buckets.long()
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))
        main_relative_pos_embeddings = torch.gather(rel_pos_embeddings, dim=1, index=main_relative_position_buckets)
        main_relative_pos_embeddings = main_relative_pos_embeddings.view(batch_size, num_attn_heads, tgt_len, -1)
        return main_relative_pos_embeddings

    def get_predict_relative_pos_embeddings(self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets):
        batch_size, sequence_length = hidden_states.shape[0:2]
        if predict_relative_position_buckets is None:
            key_sequence_length = attn_weights.shape[-1]
            assert position_ids[0][0] == key_sequence_length - 1, '`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)'
            relative_positions = torch.arange(0, key_sequence_length).unsqueeze(0).unsqueeze(0).repeat(batch_size, sequence_length, 1).to(position_ids.device)
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            predict_relative_position_buckets = compute_relative_buckets(self.num_buckets, self.relative_max_distance, relative_positions, False)
        hidden_states = hidden_states.transpose(1, 2)
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
        rel_pos_embeddings = rel_pos_embeddings.view(hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads))
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 2, 1, 4, 3)
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, self.num_buckets)
        predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0)
        predict_relative_position_buckets = predict_relative_position_buckets.repeat(self.ngram, 1, self.num_attn_heads, 1)
        predict_relative_position_buckets = predict_relative_position_buckets.view(-1, predict_relative_position_buckets.size(-1)).long()
        predict_relative_pos_embeddings = torch.gather(rel_pos_embeddings, dim=1, index=predict_relative_position_buckets)
        predict_relative_pos_embeddings = predict_relative_pos_embeddings.view(batch_size, self.ngram, self.num_attn_heads, sequence_length, -1)
        return predict_relative_pos_embeddings