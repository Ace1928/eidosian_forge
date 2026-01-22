import math
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_distilbert import DistilBertConfig
class FlaxFFN(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dropout = nn.Dropout(rate=self.config.dropout)
        self.chunk_size_feed_forward = self.config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Dense(self.config.hidden_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.lin2 = nn.Dense(self.config.dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.activation = ACT2FN[self.config.activation]

    def __call__(self, hidden_states, deterministic: bool=True):
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.lin2(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states