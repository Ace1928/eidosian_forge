import math
import random
from functools import partial
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, logging, replace_return_docstrings
from .configuration_pegasus import PegasusConfig
class FlaxPegasusModule(nn.Module):
    config: PegasusConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.shared = nn.Embed(self.config.vocab_size, self.config.d_model, embedding_init=jax.nn.initializers.normal(self.config.init_std), dtype=self.dtype)
        self.encoder = FlaxPegasusEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        self.decoder = FlaxPegasusDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, position_ids=decoder_position_ids, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return FlaxSeq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)