import copy
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_t5 import T5Config
@add_start_docstrings("The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.", T5_START_DOCSTRING)
class FlaxT5EncoderModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.shared = nn.Embed(self.config.vocab_size, self.config.d_model, embedding_init=jax.nn.initializers.normal(self.config.initializer_factor * 1.0), dtype=self.dtype)
        encoder_config = copy.deepcopy(self.config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.causal = False
        self.encoder = FlaxT5Stack(encoder_config, embed_tokens=self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)

    def __call__(self, input_ids=None, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict: bool=True, deterministic: bool=True):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        return encoder_outputs