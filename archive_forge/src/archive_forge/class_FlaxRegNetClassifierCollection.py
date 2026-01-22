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
class FlaxRegNetClassifierCollection(nn.Module):
    config: RegNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype, name='1')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.classifier(x)