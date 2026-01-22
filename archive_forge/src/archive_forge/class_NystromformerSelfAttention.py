import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_nystromformer import NystromformerConfig
class NystromformerSelfAttention(nn.Module):

    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_landmarks = config.num_landmarks
        self.seq_len = config.segment_means_seq_len
        self.conv_kernel_size = config.conv_kernel_size
        if config.inv_coeff_init_option:
            self.init_option = config['inv_init_coeff_option']
        else:
            self.init_option = 'original'
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, 'position_embedding_type', 'absolute')
        if self.conv_kernel_size is not None:
            self.conv = nn.Conv2d(in_channels=self.num_attention_heads, out_channels=self.num_attention_heads, kernel_size=(self.conv_kernel_size, 1), padding=(self.conv_kernel_size // 2, 0), bias=False, groups=self.num_attention_heads)

    def iterative_inv(self, mat, n_iter=6):
        identity = torch.eye(mat.size(-1), device=mat.device)
        key = mat
        if self.init_option == 'original':
            value = 1 / torch.max(torch.sum(key, dim=-2)) * key.transpose(-1, -2)
        else:
            value = 1 / torch.max(torch.sum(key, dim=-2), dim=-1).values[:, :, None, None] * key.transpose(-1, -2)
        for _ in range(n_iter):
            key_value = torch.matmul(key, value)
            value = torch.matmul(0.25 * value, 13 * identity - torch.matmul(key_value, 15 * identity - torch.matmul(key_value, 7 * identity - key_value)))
        return value

    def transpose_for_scores(self, layer):
        new_layer_shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        layer = layer.view(*new_layer_shape)
        return layer.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer / math.sqrt(math.sqrt(self.attention_head_size))
        key_layer = key_layer / math.sqrt(math.sqrt(self.attention_head_size))
        if self.num_landmarks == self.seq_len:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            context_layer = torch.matmul(attention_probs, value_layer)
        else:
            q_landmarks = query_layer.reshape(-1, self.num_attention_heads, self.num_landmarks, self.seq_len // self.num_landmarks, self.attention_head_size).mean(dim=-2)
            k_landmarks = key_layer.reshape(-1, self.num_attention_heads, self.num_landmarks, self.seq_len // self.num_landmarks, self.attention_head_size).mean(dim=-2)
            kernel_1 = torch.nn.functional.softmax(torch.matmul(query_layer, k_landmarks.transpose(-1, -2)), dim=-1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2)), dim=-1)
            attention_scores = torch.matmul(q_landmarks, key_layer.transpose(-1, -2))
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            kernel_3 = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = torch.matmul(kernel_1, self.iterative_inv(kernel_2))
            new_value_layer = torch.matmul(kernel_3, value_layer)
            context_layer = torch.matmul(attention_probs, new_value_layer)
        if self.conv_kernel_size is not None:
            context_layer += self.conv(value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs