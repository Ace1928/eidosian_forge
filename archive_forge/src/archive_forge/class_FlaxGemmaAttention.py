from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_gemma import GemmaConfig
class FlaxGemmaAttention(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    is_cross_attention: bool = False

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        kernel = jax.nn.initializers.normal(self.config.initializer_range)
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=config.attention_bias, dtype=self.dtype, kernel_init=kernel)
        self.k_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=config.attention_bias, dtype=self.dtype, kernel_init=kernel)
        self.v_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=config.attention_bias, dtype=self.dtype, kernel_init=kernel)
        self.o_proj = nn.Dense(self.embed_dim, use_bias=config.attention_bias, dtype=self.dtype, kernel_init=kernel)
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype='bool'), dtype='bool')
        self.rotary_emb = FlaxGemmaRotaryEmbedding(config, dtype=self.dtype)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads * self.head_dim,))

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

    def __call__(self, hidden_states, attention_mask, position_ids, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = self._split_heads(query, self.num_heads)
        key = self._split_heads(key, self.num_key_value_heads)
        value = self._split_heads(value, self.num_key_value_heads)
        key, query = self.rotary_emb(key, query, position_ids)
        query_length, key_length = (query.shape[1], key.shape[1])
        if self.has_variable('cache', 'cached_key'):
            mask_shift = self.variables['cache']['cache_index']
            max_decoder_length = self.variables['cache']['cached_key'].shape[1]
            causal_mask = lax.dynamic_slice(self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length))
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)
        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng('dropout')
        if self.has_variable('cache', 'cached_key') or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)
        attention_bias = lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype))
        key = jnp.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, repeats=self.num_key_value_groups, axis=2)
        attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
        attn_weights = dot_product_attention_weights(query, key, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.config.attention_dropout, deterministic=deterministic, dtype=attention_dtype)
        if self.attention_softmax_in_fp32:
            attn_weights = attn_weights.astype(self.dtype)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs