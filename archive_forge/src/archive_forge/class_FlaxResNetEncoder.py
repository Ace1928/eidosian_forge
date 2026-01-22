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
class FlaxResNetEncoder(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.stages = FlaxResNetStageCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_state: jnp.ndarray, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True) -> FlaxBaseModelOutputWithNoAttention:
        hidden_state, hidden_states = self.stages(hidden_state, output_hidden_states=output_hidden_states, deterministic=deterministic)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)
        if not return_dict:
            return tuple((v for v in [hidden_state, hidden_states] if v is not None))
        return FlaxBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)