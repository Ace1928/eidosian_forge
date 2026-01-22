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
class FlaxTransformerBlock(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.config.dim % self.config.n_heads == 0, f'Hidden size {self.config.dim} not dividable by number of heads {self.config.n_heads}'
        self.attention = FlaxMultiHeadSelfAttention(self.config, dtype=self.dtype)
        self.sa_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)
        self.ffn = FlaxFFN(self.config, dtype=self.dtype)
        self.output_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)

    def __call__(self, hidden_states, attn_mask, output_attentions: bool=False, deterministic: bool=True):
        sa_output = self.attention(query=hidden_states, key=hidden_states, value=hidden_states, mask=attn_mask, output_attentions=output_attentions, deterministic=deterministic)
        if output_attentions:
            sa_output, sa_weights = sa_output
        else:
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + hidden_states)
        ffn_output = self.ffn(sa_output, deterministic=deterministic)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)
        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output