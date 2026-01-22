import math
import random
from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...generation.flax_logits_process import FlaxWhisperTimeStampLogitsProcessor
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
class FlaxWhisperEncoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.conv1 = nn.Conv(self.config.d_model, kernel_size=(3,), padding=1, kernel_init=jax.nn.initializers.normal(self.config.init_std), dtype=self.dtype)
        self.conv2 = nn.Conv(self.config.d_model, kernel_size=(3,), strides=2, padding=1, kernel_init=jax.nn.initializers.normal(self.config.init_std), dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.layers = FlaxWhisperEncoderLayerCollection(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.embed_positions = nn.Embed(self.config.max_source_positions, self.config.d_model, dtype=self.dtype, embedding_init=sinusoidal_embedding_init)
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(self, input_features: jnp.ndarray, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True) -> Tuple[jnp.ndarray]:
        if input_features.shape[1:] != (self.config.num_mel_bins, self.config.max_source_positions * 2):
            raise ValueError(f'input_features.shape[1:], must be equal to (self.config.num_mel_bins, self.config.max_source_positions * 2) (got {input_features.shape[1:]}, but should be ({self.config.num_mel_bins}, {self.config.max_source_positions * 2}))')
        input_features = input_features.transpose(0, 2, 1)
        hidden_states = jax.nn.gelu(self.conv1(input_features), approximate=False)
        hidden_states = jax.nn.gelu(self.conv2(hidden_states), approximate=False)
        embed_positions = self.embed_positions(jnp.arange(self.config.max_source_positions))
        embed_positions = jax.lax.stop_gradient(embed_positions)
        hidden_states = hidden_states + embed_positions
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        outputs = self.layers(hidden_states, attention_mask=None, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutput(last_hidden_state=last_hidden_states, hidden_states=hidden_states, attentions=outputs.attentions)