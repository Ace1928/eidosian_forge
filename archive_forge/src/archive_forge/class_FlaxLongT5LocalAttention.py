import copy
from typing import Any, Callable, List, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_longt5 import LongT5Config
class FlaxLongT5LocalAttention(nn.Module):
    config: LongT5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.relative_attention_num_buckets = self.config.relative_attention_num_buckets
        self.relative_attention_max_distance = self.config.relative_attention_max_distance
        self.d_model = self.config.d_model
        self.key_value_proj_dim = self.config.d_kv
        self.n_heads = self.config.num_heads
        self.local_radius = self.config.local_radius
        self.block_len = self.local_radius + 1
        self.dropout = self.config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        q_init_std = self.config.initializer_factor * (self.inner_dim * self.key_value_proj_dim) ** (-0.5)
        kv_init_std = self.config.initializer_factor * self.inner_dim ** (-0.5)
        o_init_std = self.config.initializer_factor * self.inner_dim ** (-0.5)
        self.q = nn.Dense(self.inner_dim, use_bias=False, kernel_init=jax.nn.initializers.normal(q_init_std), dtype=self.dtype)
        self.k = nn.Dense(self.inner_dim, use_bias=False, kernel_init=jax.nn.initializers.normal(kv_init_std), dtype=self.dtype)
        self.v = nn.Dense(self.inner_dim, use_bias=False, kernel_init=jax.nn.initializers.normal(kv_init_std), dtype=self.dtype)
        self.o = nn.Dense(self.d_model, use_bias=False, kernel_init=jax.nn.initializers.normal(o_init_std), dtype=self.dtype)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embed(self.relative_attention_num_buckets, self.n_heads, embedding_init=jax.nn.initializers.normal(kv_init_std))

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, a_max=0)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_position_if_large = jnp.clip(relative_position_if_large, a_max=num_buckets - 1)
        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets.astype('i4')

    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        memory_position = jnp.arange(3 * block_length, dtype='i4')
        context_position = memory_position[block_length:-block_length]
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=True, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, None, :, :, :]
        return values

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.n_heads, self.key_value_proj_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[0], -1, self.inner_dim)

    def _create_position_bias(self, block_len: int, attention_mask: Optional[np.ndarray]) -> np.ndarray:
        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(block_len)
        elif attention_mask is not None:
            position_bias = jnp.zeros_like(attention_mask)
        else:
            position_bias = jnp.zeros((1, 1, self.n_heads, block_len, 3 * block_len), dtype=self.dtype)
        return position_bias

    def __call__(self, hidden_states, attention_mask=None, key_value_states=None, position_bias=None, output_attentions=False, deterministic=True):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        batch_size, seq_length = hidden_states.shape[:2]
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states) if key_value_states is None else self.k(key_value_states)
        value_states = self.v(hidden_states) if key_value_states is None else self.v(key_value_states)
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)
        query_states = _split_into_blocks(query_states, self.block_len, axis=1)
        key_states = _split_into_blocks(key_states, self.block_len, axis=1)
        value_states = _split_into_blocks(value_states, self.block_len, axis=1)
        key_states = _concatenate_3_blocks(key_states, block_axis=1, sequence_axis=2)
        value_states = _concatenate_3_blocks(value_states, block_axis=1, sequence_axis=2)
        query_states *= jnp.sqrt(query_states.shape[-1])
        if attention_mask is not None:
            attention_mask = _get_local_attention_mask(attention_mask, self.block_len)
            attention_mask = jax.lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, -10000000000.0).astype(self.dtype))
        if position_bias is None:
            position_bias = self._create_position_bias(self.block_len, attention_mask)
            if attention_mask is not None:
                position_bias = position_bias + attention_mask.swapaxes(1, 2)
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng('dropout')
        attn_weights = dot_product_attention_weights(query_states, key_states, bias=position_bias, dropout_rng=dropout_rng, dropout_rate=self.dropout, broadcast_dropout=True, deterministic=deterministic, dtype=self.dtype)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = attn_output[:, :seq_length, :]
        attn_output = self.o(attn_output)
        outputs = (attn_output, position_bias)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs