from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_utils import FlaxPreTrainedModel, append_replace_return_docstrings, overwrite_call_docstring
from ...utils import add_start_docstrings, logging
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_flax_auto import FLAX_MODEL_MAPPING, FlaxAutoModel
from ..clip.modeling_flax_clip import FlaxCLIPOutput, FlaxCLIPVisionModel
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig
def _get_features(module, pixel_values, deterministic):
    vision_outputs = module.vision_model(pixel_values=pixel_values, deterministic=deterministic)
    pooled_output = vision_outputs[1]
    image_features = module.visual_projection(pooled_output)
    return image_features