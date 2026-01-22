from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_electra import ElectraConfig
class FlaxElectraAttention(nn.Module):
    config: ElectraConfig
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self = FlaxElectraSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        self.output = FlaxElectraSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, layer_head_mask, key_value_states=None, init_cache=False, deterministic=True, output_attentions: bool=False):
        attn_outputs = self.self(hidden_states, attention_mask, layer_head_mask=layer_head_mask, key_value_states=key_value_states, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        return outputs