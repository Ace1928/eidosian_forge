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
from .configuration_bart import BartConfig
class FlaxBartForConditionalGenerationModule(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.model = FlaxBartModule(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(self.model.shared.num_embeddings, use_bias=False, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std))
        self.final_logits_bias = self.param('final_logits_bias', self.bias_init, (1, self.model.shared.num_embeddings))

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, position_ids=position_ids, decoder_position_ids=decoder_position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.model.variables['params']['shared']['embedding']
            lm_logits = self.lm_head.apply({'params': {'kernel': shared_embedding.T}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
        lm_logits += jax.lax.stop_gradient(self.final_logits_bias.astype(self.dtype))
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output
        return FlaxSeq2SeqLMOutput(logits=lm_logits, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)