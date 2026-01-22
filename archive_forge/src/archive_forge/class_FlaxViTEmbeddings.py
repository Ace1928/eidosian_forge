from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_vit import ViTConfig
class FlaxViTEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.cls_token = self.param('cls_token', jax.nn.initializers.variance_scaling(self.config.initializer_range ** 2, 'fan_in', 'truncated_normal'), (1, 1, self.config.hidden_size))
        self.patch_embeddings = FlaxViTPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param('position_embeddings', jax.nn.initializers.variance_scaling(self.config.initializer_range ** 2, 'fan_in', 'truncated_normal'), (1, num_patches + 1, self.config.hidden_size))
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, pixel_values, deterministic=True):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings