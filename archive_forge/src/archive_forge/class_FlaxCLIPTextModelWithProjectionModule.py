from typing import Any, Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
class FlaxCLIPTextModelWithProjectionModule(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.text_model = FlaxCLIPTextTransformer(self.config, dtype=self.dtype)
        self.text_projection = nn.Dense(self.config.projection_dim, use_bias=False, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, position_ids, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = text_outputs[1]
        text_embeds = self.text_projection(pooled_output)
        if not return_dict:
            return (text_embeds, text_outputs[0]) + text_outputs[2:]
        return FlaxCLIPTextModelOutput(text_embeds=text_embeds, last_hidden_state=text_outputs.last_hidden_state, hidden_states=text_outputs.hidden_states, attentions=text_outputs.attentions)