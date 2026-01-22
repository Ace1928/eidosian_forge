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
class FlaxElectraModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.embeddings = FlaxElectraEmbeddings(self.config, dtype=self.dtype)
        if self.config.embedding_size != self.config.hidden_size:
            self.embeddings_project = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.encoder = FlaxElectraEncoder(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask: Optional[np.ndarray]=None, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        embeddings = self.embeddings(input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic)
        if hasattr(self, 'embeddings_project'):
            embeddings = self.embeddings_project(embeddings)
        return self.encoder(embeddings, attention_mask, head_mask=head_mask, deterministic=deterministic, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)