import gc
import json
import os
import re
import warnings
from functools import partial
from pickle import UnpicklingError
from typing import Any, Dict, Optional, Set, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import FlaxGenerationMixin, GenerationConfig
from .modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from .utils.import_utils import is_safetensors_available
def to_bf16(self, params: Union[Dict, FrozenDict], mask: Any=None):
    """
        Cast the floating-point `params` to `jax.numpy.bfloat16`. This returns a new `params` tree and does not cast
        the `params` in place.

        This method can be used on TPU to explicitly convert the model parameters to bfloat16 precision to do full
        half-precision training or to save weights in bfloat16 for inference in order to save memory and improve speed.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip.

        Examples:

        ```python
        >>> from transformers import FlaxBertModel

        >>> # load model
        >>> model = FlaxBertModel.from_pretrained("google-bert/bert-base-cased")
        >>> # By default, the model parameters will be in fp32 precision, to cast these to bfloat16 precision
        >>> model.params = model.to_bf16(model.params)
        >>> # If you want don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util

        >>> model = FlaxBertModel.from_pretrained("google-bert/bert-base-cased")
        >>> flat_params = traverse_util.flatten_dict(model.params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> model.params = model.to_bf16(model.params, mask)
        ```"""
    return self._cast_floating_to(params, jnp.bfloat16, mask)