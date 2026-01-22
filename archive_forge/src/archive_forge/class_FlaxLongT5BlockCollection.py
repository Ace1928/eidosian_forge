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
class FlaxLongT5BlockCollection(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.causal = self.config.causal
        if self.gradient_checkpointing:
            FlaxLongT5CheckpointLayer = remat(FlaxLongT5LayerCollection, static_argnums=(6, 7, 8))
            self.blocks = [FlaxLongT5CheckpointLayer(self.config, has_relative_attention_bias=i == 0, dtype=self.dtype, name=str(i)) for i in range(self.config.num_layers)]
        else:
            self.blocks = [FlaxLongT5LayerCollection(self.config, has_relative_attention_bias=i == 0, dtype=self.dtype, name=str(i)) for i in range(self.config.num_layers)]

    def __call__(self, hidden_states=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions: bool=False, output_hidden_states: bool=False, deterministic: bool=True, init_cache: bool=False):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.causal else None
        position_bias = None
        encoder_decoder_position_bias = None
        for i, layer_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, position_bias, encoder_hidden_states, encoder_attention_mask, encoder_decoder_position_bias, output_attentions, deterministic, init_cache)
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]
            if self.causal and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.causal:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)
        return FlaxBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)