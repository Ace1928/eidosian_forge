from typing import Callable, List, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig
class FlaxBeitEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.config.hidden_size))
        if self.config.use_mask_token:
            self.mask_token = self.param('mask_token', nn.initializers.zeros, (1, 1, self.config.hidden_size))
        self.patch_embeddings = FlaxBeitPatchEmbeddings(self.config, dtype=self.dtype)
        num_patches = self.patch_embeddings.num_patches
        if self.config.use_absolute_position_embeddings:
            self.position_embeddings = self.param('position_embeddings', nn.initializers.zeros, (1, num_patches + 1, self.config.hidden_size))
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, pixel_values, bool_masked_pos=None, deterministic=True):
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.shape
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        cls_tokens = cls_tokens.astype(embeddings.dtype)
        if bool_masked_pos is not None:
            mask_tokens = jnp.broadcast_to(self.mask_token, (batch_size, seq_len, self.config.hidden_size))
            mask_tokens = mask_tokens.astype(embeddings.dtype)
            w = jnp.expand_dims(bool_masked_pos, axis=-1)
            embeddings = embeddings * (1 - w) + mask_tokens * w
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        if self.config.use_absolute_position_embeddings:
            embeddings = embeddings + self.position_embeddings.astype(embeddings.dtype)
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings