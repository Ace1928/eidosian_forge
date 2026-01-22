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
class FlaxRegNetSELayerCollection(nn.Module):
    in_channels: int
    reduced_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_1 = nn.Conv(self.reduced_channels, kernel_size=(1, 1), kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='truncated_normal'), dtype=self.dtype, name='0')
        self.conv_2 = nn.Conv(self.in_channels, kernel_size=(1, 1), kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='truncated_normal'), dtype=self.dtype, name='2')

    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        hidden_state = self.conv_1(hidden_state)
        hidden_state = nn.relu(hidden_state)
        hidden_state = self.conv_2(hidden_state)
        attention = nn.sigmoid(hidden_state)
        return attention