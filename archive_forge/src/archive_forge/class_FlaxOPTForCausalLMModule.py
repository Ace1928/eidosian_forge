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
@add_start_docstrings('The bare OPT Model transformer outputting raw hidden-states without any specific head on top.', OPT_START_DOCSTRING)
class FlaxOPTForCausalLMModule(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = FlaxOPTModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std))

    def __call__(self, input_ids, attention_mask, position_ids, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        outputs = self.model(input_ids, attention_mask, position_ids, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables['params']['decoder']['embed_tokens']['embedding']
            lm_logits = self.lm_head.apply({'params': {'kernel': shared_embedding.T}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
        if not return_dict:
            return (lm_logits,) + outputs[1:]
        return FlaxMaskedLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)