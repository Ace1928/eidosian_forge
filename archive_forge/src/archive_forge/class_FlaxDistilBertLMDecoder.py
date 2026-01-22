import math
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_distilbert import DistilBertConfig
class FlaxDistilBertLMDecoder(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.bias = self.param('bias', self.bias_init, (self.config.vocab_size,))

    def __call__(self, inputs, kernel):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        y = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        bias = jnp.asarray(self.bias, self.dtype)
        y = y + bias
        return y