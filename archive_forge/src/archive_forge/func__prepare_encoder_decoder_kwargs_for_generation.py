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
def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, params, model_kwargs):
    encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not (argument.startswith('decoder_') or argument.startswith('cross_attn'))}
    model_kwargs['encoder_outputs'] = self.encode(input_ids, params=params, return_dict=True, **encoder_kwargs)
    return model_kwargs