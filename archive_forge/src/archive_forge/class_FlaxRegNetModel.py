from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import RegNetConfig
from transformers.modeling_flax_outputs import (
from transformers.modeling_flax_utils import (
from transformers.utils import (
@add_start_docstrings('The bare RegNet model outputting raw features without any specific head on top.', REGNET_START_DOCSTRING)
class FlaxRegNetModel(FlaxRegNetPreTrainedModel):
    module_class = FlaxRegNetModule