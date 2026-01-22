import math
from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention_weights, make_causal_mask
from flax.linen.activation import tanh
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_bloom import BloomConfig
class FlaxBloomBlock(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.input_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.self_attention = FlaxBloomAttention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.mlp = FlaxBloomMLP(self.config, dtype=self.dtype)
        self.apply_residual_connection_post_layernorm = self.config.apply_residual_connection_post_layernorm
        self.hidden_dropout = self.config.hidden_dropout

    def __call__(self, hidden_states, alibi, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        layernorm_output = self.input_layernorm(hidden_states)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        attn_outputs = self.self_attention(layernorm_output, residual=residual, alibi=alibi, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        post_layernorm = self.post_attention_layernorm(attention_output)
        if self.apply_residual_connection_post_layernorm:
            residual = post_layernorm
        else:
            residual = attention_output
        output = self.mlp(post_layernorm, residual, deterministic=deterministic)
        outputs = (output,) + outputs
        return outputs