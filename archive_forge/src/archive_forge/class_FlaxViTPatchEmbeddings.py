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
class FlaxViTPatchEmbeddings(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_patches = image_size // patch_size * (image_size // patch_size)
        self.num_patches = num_patches
        self.num_channels = self.config.num_channels
        self.projection = nn.Conv(self.config.hidden_size, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size), padding='VALID', dtype=self.dtype, kernel_init=jax.nn.initializers.variance_scaling(self.config.initializer_range ** 2, 'fan_in', 'truncated_normal'))

    def __call__(self, pixel_values):
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        embeddings = self.projection(pixel_values)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))