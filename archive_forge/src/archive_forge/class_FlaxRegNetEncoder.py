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
class FlaxRegNetEncoder(nn.Module):
    config: RegNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.stages = FlaxRegNetStageCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_state: jnp.ndarray, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True) -> FlaxBaseModelOutputWithNoAttention:
        hidden_state, hidden_states = self.stages(hidden_state, output_hidden_states=output_hidden_states, deterministic=deterministic)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)
        if not return_dict:
            return tuple((v for v in [hidden_state, hidden_states] if v is not None))
        return FlaxBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)