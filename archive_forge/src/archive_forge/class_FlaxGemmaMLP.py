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
class FlaxGemmaMLP(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.act = ACT2FN[self.config.hidden_act]
        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.down_proj = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)

    def __call__(self, hidden_states):
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))
        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states