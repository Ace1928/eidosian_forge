import math
import random
from functools import partial
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bart import BartConfig
class FlaxBartForSequenceClassificationModule(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32
    num_labels: Optional[int] = None

    def setup(self):
        self.model = FlaxBartModule(config=self.config, dtype=self.dtype)
        self.classification_head = FlaxBartClassificationHead(config=self.config, inner_dim=self.config.d_model, num_classes=self.num_labels if self.num_labels is not None else self.config.num_labels, pooler_dropout=self.config.classifier_dropout)

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, position_ids=position_ids, decoder_position_ids=decoder_position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        hidden_states = outputs[0]
        eos_mask = jnp.where(input_ids == self.config.eos_token_id, 1, 0)
        if type(eos_mask) != jax.interpreters.partial_eval.DynamicJaxprTracer:
            if len(jnp.unique(eos_mask.sum(1))) > 1:
                raise ValueError('All examples must have the same number of <eos> tokens.')
            if any(eos_mask.sum(1) == 0):
                raise ValueError('There are missing <eos> tokens in input_ids')
            eos_mask_noised = eos_mask + jnp.arange(eos_mask.shape[1]) * 1e-06
            eos_mask = jnp.where(eos_mask_noised == eos_mask_noised.max(1).reshape(-1, 1), 1, 0)
        sentence_representation = jnp.einsum('ijk, ij -> ijk', hidden_states, eos_mask).sum(1)
        logits = self.classification_head(sentence_representation, deterministic=deterministic)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return output
        return FlaxSeq2SeqSequenceClassifierOutput(logits=logits, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)