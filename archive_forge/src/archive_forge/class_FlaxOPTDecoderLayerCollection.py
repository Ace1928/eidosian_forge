from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxMaskedLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, logging
from .configuration_opt import OPTConfig
class FlaxOPTDecoderLayerCollection(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [FlaxOPTDecoderLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]
        self.layerdrop = self.config.layerdrop

    def __call__(self, hidden_states, attention_mask, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, init_cache=init_cache, output_attentions=output_attentions, deterministic=deterministic)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        outputs = [hidden_states, all_hidden_states, all_self_attns]
        return outputs