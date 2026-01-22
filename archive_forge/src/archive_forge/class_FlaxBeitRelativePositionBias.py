from typing import Callable, List, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig
class FlaxBeitRelativePositionBias(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        num_relative_distance = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) + 3
        self.relative_position_bias_table = self.param('relative_position_bias_table', nn.initializers.zeros, (num_relative_distance, self.config.num_attention_heads))
        self.relative_position_index = relative_position_index_init(self.window_size)

    def __call__(self):
        index = self.relative_position_index.reshape(-1)
        shape = (self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
        relative_position_bias = self.relative_position_bias_table[index].reshape(shape)
        return jnp.transpose(relative_position_bias, (2, 0, 1))