from __future__ import annotations
import itertools
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_flaubert import FlaubertConfig
@add_start_docstrings('\n    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', FLAUBERT_START_DOCSTRING)
class TFFlaubertWithLMHeadModel(TFFlaubertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFFlaubertMainLayer(config, name='transformer')
        self.pred_layer = TFFlaubertPredLayer(config, self.transformer.embeddings, name='pred_layer_._proj')
        self.supports_xla_generation = False

    def get_lm_head(self):
        return self.pred_layer

    def get_prefix_bias_name(self):
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.pred_layer.name

    def prepare_inputs_for_generation(self, inputs, **kwargs):
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id
        effective_batch_size = inputs.shape[0]
        mask_token = tf.fill((effective_batch_size, 1), 1) * mask_token_id
        inputs = tf.concat([inputs, mask_token], axis=1)
        if lang_id is not None:
            langs = tf.ones_like(inputs) * lang_id
        else:
            langs = None
        return {'input_ids': inputs, 'langs': langs}

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFFlaubertWithLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, langs: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, lengths: np.ndarray | tf.Tensor | None=None, cache: Optional[Dict[str, tf.Tensor]]=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[Tuple, TFFlaubertWithLMHeadModelOutput]:
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        output = transformer_outputs[0]
        outputs = self.pred_layer(output)
        if not return_dict:
            return (outputs,) + transformer_outputs[1:]
        return TFFlaubertWithLMHeadModelOutput(logits=outputs, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'transformer', None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, 'pred_layer', None) is not None:
            with tf.name_scope(self.pred_layer.name):
                self.pred_layer.build(None)