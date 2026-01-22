from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_albert import AlbertConfig
class FlaxAlbertLayerCollections(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32
    layer_index: Optional[str] = None

    def setup(self):
        self.albert_layers = FlaxAlbertLayerCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False):
        outputs = self.albert_layers(hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        return outputs