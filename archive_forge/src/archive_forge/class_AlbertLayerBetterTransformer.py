from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from .base import BetterTransformerBaseLayer
class AlbertLayerBetterTransformer(BetterTransformerBaseLayer, nn.Module):

    def __init__(self, albert_layer, config):
        """
        A simple conversion of the ALBERT layer to its `BetterTransformer` implementation.

        Args:
            albert_layer (`torch.nn.Module`):
                The original ALBERT Layer where the weights needs to be retrieved.
        """
        super().__init__(config)
        super(BetterTransformerBaseLayer, self).__init__()
        self.in_proj_weight = nn.Parameter(torch.cat([albert_layer.attention.query.weight, albert_layer.attention.key.weight, albert_layer.attention.value.weight]))
        self.in_proj_bias = nn.Parameter(torch.cat([albert_layer.attention.query.bias, albert_layer.attention.key.bias, albert_layer.attention.value.bias]))
        self.out_proj_weight = albert_layer.attention.dense.weight
        self.out_proj_bias = albert_layer.attention.dense.bias
        self.linear1_weight = albert_layer.ffn.weight
        self.linear1_bias = albert_layer.ffn.bias
        self.linear2_weight = albert_layer.ffn_output.weight
        self.linear2_bias = albert_layer.ffn_output.bias
        self.norm1_eps = albert_layer.attention.LayerNorm.eps
        self.norm1_weight = albert_layer.attention.LayerNorm.weight
        self.norm1_bias = albert_layer.attention.LayerNorm.bias
        self.norm2_eps = albert_layer.full_layer_layer_norm.eps
        self.norm2_weight = albert_layer.full_layer_layer_norm.weight
        self.norm2_bias = albert_layer.full_layer_layer_norm.bias
        self.num_heads = albert_layer.attention.num_attention_heads
        self.embed_dim = albert_layer.attention.all_head_size
        self.is_last_layer = False
        self.original_layers_mapping = {'in_proj_weight': ['attention.query.weight', 'attention.key.weight', 'attention.value.weight'], 'in_proj_bias': ['attention.query.bias', 'attention.key.bias', 'attention.value.bias'], 'out_proj_weight': 'attention.dense.weight', 'out_proj_bias': 'attention.dense.bias', 'linear1_weight': 'ffn.weight', 'linear1_bias': 'ffn.bias', 'linear2_weight': 'ffn_output.weight', 'linear2_bias': 'ffn_output.bias', 'norm1_eps': 'attention.LayerNorm.eps', 'norm1_weight': 'attention.LayerNorm.weight', 'norm1_bias': 'attention.LayerNorm.bias', 'norm2_eps': 'full_layer_layer_norm.eps', 'norm2_weight': 'full_layer_layer_norm.weight', 'norm2_bias': 'full_layer_layer_norm.bias'}
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.act_fn_callable = ACT2FN[self.act_fn]
        self.validate_bettertransformer()

    def forward(self, hidden_states, attention_mask, *_):
        if not self.training and (not torch.is_autocast_enabled()) and (not torch.is_autocast_cpu_enabled()):
            if hidden_states.is_nested:
                attention_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
                attention_mask = None
            hidden_states = torch._transformer_encoder_layer_fwd(hidden_states, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj_weight, self.out_proj_bias, self.use_gelu, self.norm_first, self.norm1_eps, self.norm1_weight, self.norm1_bias, self.norm2_weight, self.norm2_bias, self.linear1_weight, self.linear1_bias, self.linear2_weight, self.linear2_bias, attention_mask)
            if hidden_states.is_nested and self.is_last_layer:
                hidden_states = hidden_states.to_padded_tensor(0.0)
        else:
            qkv = F.linear(hidden_states, weight=self.in_proj_weight, bias=self.in_proj_bias)
            qkv = qkv.view(qkv.size()[:-1] + (3, self.num_heads, self.attention_head_size)).permute(2, 0, 3, 1, 4)
            query, key, value = (qkv[0], qkv[1], qkv[2])
            if self.training:
                attention_mask = None
            attention_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False, dropout_p=self.attention_probs_dropout_prob if self.training else 0.0)
            attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
            new_attention_out_shape = attention_out.size()[:-2] + (self.num_heads * self.attention_head_size,)
            attention_out = attention_out.view(new_attention_out_shape)
            attention_out = F.layer_norm(F.dropout(F.linear(attention_out, self.out_proj_weight, self.out_proj_bias), p=self.hidden_dropout_prob, training=self.training) + hidden_states, normalized_shape=self.norm1_weight.shape, weight=self.norm1_weight, bias=self.norm1_bias)
            hidden_states = self.act_fn_callable(F.linear(attention_out, self.linear1_weight, self.linear1_bias))
            hidden_states = F.layer_norm(attention_out + F.dropout(F.linear(hidden_states, self.linear2_weight, self.linear2_bias), p=self.hidden_dropout_prob, training=self.training), normalized_shape=self.norm2_weight.shape, weight=self.norm2_weight, bias=self.norm2_bias)
        return (hidden_states,)