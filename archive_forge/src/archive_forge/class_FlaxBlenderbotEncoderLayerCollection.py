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
class FlaxBlenderbotEncoderLayerCollection(nn.Module):
    config: BlenderbotConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [FlaxBlenderbotEncoderLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.encoder_layers)]
        self.layerdrop = self.config.encoder_layerdrop

    def __call__(self, hidden_states, attention_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if not deterministic and dropout_probability < self.layerdrop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask, output_attentions, deterministic)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        outputs = (hidden_states, all_hidden_states, all_attentions)
        if not return_dict:
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)