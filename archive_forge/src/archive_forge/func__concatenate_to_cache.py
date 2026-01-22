from functools import partial
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
from .configuration_llama import LlamaConfig
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