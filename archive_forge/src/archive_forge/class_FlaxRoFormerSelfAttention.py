from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_roformer import RoFormerConfig
class FlaxRoFormerSelfAttention(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError('`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`                    : {self.config.num_attention_heads}')
        self.query = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.key = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.value = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.rotary_value = self.config.rotary_value

    def __call__(self, hidden_states, attention_mask, sinusoidal_pos, layer_head_mask, deterministic=True, output_attentions: bool=False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        query_states = self.query(hidden_states).reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim))
        value_states = self.value(hidden_states).reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim))
        key_states = self.key(hidden_states).reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim))
        if sinusoidal_pos is not None:
            if self.rotary_value:
                query_states, key_states, value_states = self.apply_rotary_position_embeddings(sinusoidal_pos, query_states, key_states, value_states)
            else:
                query_states, key_states = self.apply_rotary_position_embeddings(sinusoidal_pos, query_states, key_states)
        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype))
        else:
            attention_bias = None
        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng('dropout')
        attn_weights = dot_product_attention_weights(query_states, key_states, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.config.attention_probs_dropout_prob, broadcast_dropout=True, deterministic=deterministic, dtype=self.dtype, precision=None)
        if layer_head_mask is not None:
            attn_weights = jnp.einsum('...hqk,h->...hqk', attn_weights, layer_head_mask)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        sin, cos = sinusoidal_pos.split(2, axis=-1)
        sin_pos = jnp.stack([sin, sin], axis=-1).reshape(sinusoidal_pos.shape)
        cos_pos = jnp.stack([cos, cos], axis=-1).reshape(sinusoidal_pos.shape)

        def rotate_layer(layer, sin_pos, cos_pos):
            rotate_half_layer = jnp.stack([-layer[..., 1::2], layer[..., ::2]], axis=-1).reshape(layer.shape)
            rotary_matrix_cos = jnp.einsum('bslh,...sh->bslh', layer, cos_pos)
            rotary_matrix_sin = jnp.einsum('bslh,...sh->bslh', rotate_half_layer, sin_pos)
            return rotary_matrix_cos + rotary_matrix_sin
        query_layer = rotate_layer(query_layer, sin_pos, cos_pos)
        key_layer = rotate_layer(key_layer, sin_pos, cos_pos)
        if value_layer is not None:
            value_layer = rotate_layer(value_layer, sin_pos, cos_pos)
            return (query_layer, key_layer, value_layer)
        return (query_layer, key_layer)