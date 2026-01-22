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
class FlaxBloomAttention(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'`hidden_size` must be divisible by `num_heads` (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        dense = partial(nn.Dense, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.query_key_value = dense(self.hidden_size * 3)
        self.dense = dense(self.hidden_size)
        self.resid_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:-1] + (self.num_heads, self.head_dim * 3))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        is_initialized = self.has_variable('cache', 'cached_key')
        cached_key = self.variable('cache', 'cached_key', jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable('cache', 'cached_value', jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable('cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32))
        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            pad_mask = jnp.broadcast_to(jnp.arange(max_length) < cur_index + num_updated_cache_vectors, tuple(batch_dims) + (1, num_updated_cache_vectors, max_length))
            attention_mask = combine_masks(pad_mask, attention_mask)
        return (key, value, attention_mask)

    def __call__(self, hidden_states, residual, alibi, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        batch_size, seq_length = hidden_states.shape[:2]
        fused_qkv = self.query_key_value(hidden_states)
        fused_qkv = self._split_heads(fused_qkv)
        query, key, value = jnp.split(fused_qkv, 3, axis=-1)
        causal_attention_mask = make_causal_mask(attention_mask, dtype='bool')
        causal_attention_mask_shift = self.variables['cache']['cache_index'] if self.has_variable('cache', 'cached_key') else 0
        if self.has_variable('cache', 'cached_key'):
            max_decoder_length = self.variables['cache']['cached_key'].shape[1]
            causal_attention_mask = jax.lax.dynamic_slice(causal_attention_mask, (0, 0, causal_attention_mask_shift, 0), (1, 1, seq_length, max_decoder_length))
        causal_attention_mask = jnp.broadcast_to(causal_attention_mask, (batch_size,) + causal_attention_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_attention_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_attention_mask)
        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng('dropout')
        if self.has_variable('cache', 'cached_key') or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)
        mask_value = jnp.finfo(self.dtype).min
        attention_bias = lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, mask_value).astype(self.dtype))
        attention_bias = attention_bias + alibi
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(query, key, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.config.attention_dropout, deterministic=deterministic, dtype=attention_dtype)
        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.dense(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        attn_output = attn_output + residual
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs