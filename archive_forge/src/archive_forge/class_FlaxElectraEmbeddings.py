from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_electra import ElectraConfig
class FlaxElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.word_embeddings = nn.Embed(self.config.vocab_size, self.config.embedding_size, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.position_embeddings = nn.Embed(self.config.max_position_embeddings, self.config.embedding_size, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.token_type_embeddings = nn.Embed(self.config.type_vocab_size, self.config.embedding_size, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask, deterministic: bool=True):
        inputs_embeds = self.word_embeddings(input_ids.astype('i4'))
        position_embeds = self.position_embeddings(position_ids.astype('i4'))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype('i4'))
        hidden_states = inputs_embeds + token_type_embeddings + position_embeds
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states