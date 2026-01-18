from typing import Any, Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

        Args:
            pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained
                using [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.

        Returns:
            image_features (`jnp.ndarray` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`FlaxCLIPVisionModel`]

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, FlaxCLIPModel

        >>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="np")

        >>> image_features = model.get_image_features(**inputs)
        ```