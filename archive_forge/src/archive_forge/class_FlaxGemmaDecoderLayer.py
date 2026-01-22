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
class FlaxGemmaDecoderLayer(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.input_layernorm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)
        self.self_attn = FlaxGemmaAttention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxGemmaRMSNorm(self.config, dtype=self.dtype)
        self.mlp = FlaxGemmaMLP(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
        attn_output = outputs[0]
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,) + outputs[1:]