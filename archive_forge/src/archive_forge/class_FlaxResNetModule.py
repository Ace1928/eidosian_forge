from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_resnet import ResNetConfig
class FlaxResNetModule(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedder = FlaxResNetEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxResNetEncoder(self.config, dtype=self.dtype)
        self.pooler = partial(nn.avg_pool, padding=((0, 0), (0, 0)))

    def __call__(self, pixel_values, deterministic: bool=True, output_hidden_states: bool=False, return_dict: bool=True) -> FlaxBaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embedding_output = self.embedder(pixel_values, deterministic=deterministic)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state, window_shape=(last_hidden_state.shape[1], last_hidden_state.shape[2]), strides=(last_hidden_state.shape[1], last_hidden_state.shape[2])).transpose(0, 3, 1, 2)
        last_hidden_state = last_hidden_state.transpose(0, 3, 1, 2)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return FlaxBaseModelOutputWithPoolingAndNoAttention(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states)