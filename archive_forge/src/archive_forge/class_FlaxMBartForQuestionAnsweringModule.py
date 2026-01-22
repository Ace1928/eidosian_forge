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
from .configuration_mbart import MBartConfig
class FlaxMBartForQuestionAnsweringModule(nn.Module):
    config: MBartConfig
    dtype: jnp.dtype = jnp.float32
    num_labels = 2

    def setup(self):
        self.model = FlaxMBartModule(config=self.config, dtype=self.dtype)
        self.qa_outputs = nn.Dense(self.num_labels, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std))

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, position_ids=position_ids, decoder_position_ids=decoder_position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = jnp.split(logits, logits.shape[-1], axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return output
        return FlaxSeq2SeqQuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)