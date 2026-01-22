import copy
from typing import Any, Callable, List, Optional, Tuple
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
from .configuration_longt5 import LongT5Config
class FlaxLongT5DenseActDense(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        wi_init_std = self.config.initializer_factor * self.config.d_model ** (-0.5)
        wo_init_std = self.config.initializer_factor * self.config.d_ff ** (-0.5)
        self.wi = nn.Dense(self.config.d_ff, use_bias=False, kernel_init=jax.nn.initializers.normal(wi_init_std), dtype=self.dtype)
        self.wo = nn.Dense(self.config.d_model, use_bias=False, kernel_init=jax.nn.initializers.normal(wo_init_std), dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.act = ACT2FN[self.config.dense_act_fn]

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.wo(hidden_states)
        return hidden_states