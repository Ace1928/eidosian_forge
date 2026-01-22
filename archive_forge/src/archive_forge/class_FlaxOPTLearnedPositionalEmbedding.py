from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxMaskedLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, logging
from .configuration_opt import OPTConfig
class FlaxOPTLearnedPositionalEmbedding(nn.Embed):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def setup(self):
        self.offset = 2
        self.embedding = self.param('embedding', self.embedding_init, (self.num_embeddings + self.offset, self.features), self.param_dtype)

    def __call__(self, positions):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        return super().__call__(positions + self.offset)