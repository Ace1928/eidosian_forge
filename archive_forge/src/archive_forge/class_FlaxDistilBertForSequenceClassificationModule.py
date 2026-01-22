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
class FlaxDistilBertForSequenceClassificationModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        self.pre_classifier = nn.Dense(self.config.dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.dropout = nn.Dropout(rate=self.config.seq_classif_dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        distilbert_output = self.distilbert(input_ids, attention_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = ACT2FN['relu'](pooled_output)
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)
        if not return_dict:
            return (logits,) + distilbert_output[1:]
        return FlaxSequenceClassifierOutput(logits=logits, hidden_states=distilbert_output.hidden_states, attentions=distilbert_output.attentions)