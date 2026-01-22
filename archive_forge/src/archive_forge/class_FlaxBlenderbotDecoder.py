import math
import random
from functools import partial
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_blenderbot import BlenderbotConfig
class FlaxBlenderbotDecoder(nn.Module):
    config: BlenderbotConfig
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        embed_dim = self.config.d_model
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings
        self.embed_scale = math.sqrt(self.config.d_model) if self.config.scale_embedding else 1.0
        self.embed_positions = nn.Embed(self.config.max_position_embeddings, embed_dim, embedding_init=jax.nn.initializers.normal(self.config.init_std))
        self.layers = FlaxBlenderbotDecoderLayerCollection(self.config, self.dtype)
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(self, input_ids, attention_mask, position_ids, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        positions = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + positions
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        outputs = self.layers(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_states, hidden_states=hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)