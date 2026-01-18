import copy
import inspect
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union
import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from ..models.auto import (
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .flax_logits_process import (
def unflatten_beam_dim(tensor, batch_size, num_beams):
    """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
    if tensor.ndim == 0:
        return tensor
    return tensor.reshape((batch_size, num_beams) + tensor.shape[1:])