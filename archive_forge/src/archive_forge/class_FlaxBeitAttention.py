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
class FlaxBeitAttention(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxBeitSelfAttention(self.config, self.window_size, dtype=self.dtype)
        self.output = FlaxBeitSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, relative_position_bias=None, deterministic=True, output_attentions: bool=False):
        attn_outputs = self.attention(hidden_states, relative_position_bias, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        attn_output = self.output(attn_output, deterministic=deterministic)
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        return outputs