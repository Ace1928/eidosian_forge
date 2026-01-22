import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ibert import IBertConfig
from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear
class IBertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.quant_mode = config.quant_mode
        self.weight_bit = 8
        self.bias_bit = 32
        self.act_bit = 8
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = QuantLinear(config.hidden_size, self.all_head_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        self.key = QuantLinear(config.hidden_size, self.all_head_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        self.value = QuantLinear(config.hidden_size, self.all_head_size, bias=True, weight_bit=self.weight_bit, bias_bit=self.bias_bit, quant_mode=self.quant_mode, per_channel=True)
        self.query_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.key_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.value_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        if self.position_embedding_type != 'absolute':
            raise ValueError("I-BERT only supports 'absolute' for `config.position_embedding_type`")
        self.softmax = IntSoftmax(self.act_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, hidden_states_scaling_factor, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer, mixed_query_layer_scaling_factor = self.query(hidden_states, hidden_states_scaling_factor)
        mixed_key_layer, mixed_key_layer_scaling_factor = self.key(hidden_states, hidden_states_scaling_factor)
        mixed_value_layer, mixed_value_layer_scaling_factor = self.value(hidden_states, hidden_states_scaling_factor)
        query_layer, query_layer_scaling_factor = self.query_activation(mixed_query_layer, mixed_query_layer_scaling_factor)
        key_layer, key_layer_scaling_factor = self.key_activation(mixed_key_layer, mixed_key_layer_scaling_factor)
        value_layer, value_layer_scaling_factor = self.value_activation(mixed_value_layer, mixed_value_layer_scaling_factor)
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scale = math.sqrt(self.attention_head_size)
        attention_scores = attention_scores / scale
        if self.quant_mode:
            attention_scores_scaling_factor = query_layer_scaling_factor * key_layer_scaling_factor / scale
        else:
            attention_scores_scaling_factor = None
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs, attention_probs_scaling_factor = self.softmax(attention_scores, attention_scores_scaling_factor)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        if attention_probs_scaling_factor is not None:
            context_layer_scaling_factor = attention_probs_scaling_factor * value_layer_scaling_factor
        else:
            context_layer_scaling_factor = None
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer, context_layer_scaling_factor = self.output_activation(context_layer, context_layer_scaling_factor)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        output_scaling_factor = (context_layer_scaling_factor, attention_probs_scaling_factor) if output_attentions else (context_layer_scaling_factor,)
        return (outputs, output_scaling_factor)