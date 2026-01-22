import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_squeezebert import SqueezeBertConfig
class SqueezeBertSelfAttention(nn.Module):

    def __init__(self, config, cin, q_groups=1, k_groups=1, v_groups=1):
        """
        config = used for some things; ignored for others (work in progress...) cin = input channels = output channels
        groups = number of groups to use in conv1d layers
        """
        super().__init__()
        if cin % config.num_attention_heads != 0:
            raise ValueError(f'cin ({cin}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(cin / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=q_groups)
        self.key = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=k_groups)
        self.value = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=v_groups)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.matmul_qk = MatMulWrapper()
        self.matmul_qkv = MatMulWrapper()

    def transpose_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, W, C2] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1])
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2)

    def transpose_key_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, C2, W] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1])
        x = x.view(*new_x_shape)
        return x

    def transpose_output(self, x):
        """
        - input: [N, C1, W, C2]
        - output: [N, C, W]
        """
        x = x.permute(0, 1, 3, 2).contiguous()
        new_x_shape = (x.size()[0], self.all_head_size, x.size()[3])
        x = x.view(*new_x_shape)
        return x

    def forward(self, hidden_states, attention_mask, output_attentions):
        """
        expects hidden_states in [N, C, W] data layout.

        The attention_mask data layout is [N, W], and it does not need to be transposed.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_score = self.matmul_qk(query_layer, key_layer)
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        attention_score = attention_score + attention_mask
        attention_probs = self.softmax(attention_score)
        attention_probs = self.dropout(attention_probs)
        context_layer = self.matmul_qkv(attention_probs, value_layer)
        context_layer = self.transpose_output(context_layer)
        result = {'context_layer': context_layer}
        if output_attentions:
            result['attention_score'] = attention_score
        return result