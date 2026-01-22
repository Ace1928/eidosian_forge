from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import RegNetConfig
from transformers.modeling_flax_outputs import (
from transformers.modeling_flax_utils import (
from transformers.utils import (
class FlaxRegNetStageLayersCollection(nn.Module):
    """
    A RegNet stage composed by stacked layers.
    """
    config: RegNetConfig
    in_channels: int
    out_channels: int
    stride: int = 2
    depth: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        layer = FlaxRegNetXLayer if self.config.layer_type == 'x' else FlaxRegNetYLayer
        layers = [layer(self.config, self.in_channels, self.out_channels, stride=self.stride, dtype=self.dtype, name='0')]
        for i in range(self.depth - 1):
            layers.append(layer(self.config, self.out_channels, self.out_channels, dtype=self.dtype, name=str(i + 1)))
        self.layers = layers

    def __call__(self, x: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        hidden_state = x
        for layer in self.layers:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state