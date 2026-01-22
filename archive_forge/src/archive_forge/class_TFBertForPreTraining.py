from __future__ import annotations
import math
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
from .configuration_bert import BertConfig
@add_start_docstrings('\nBert Model with two heads on top as done during the pretraining:\n    a `masked language modeling` head and a `next sentence prediction (classification)` head.\n    ', BERT_START_DOCSTRING)
class TFBertForPreTraining(TFBertPreTrainedModel, TFBertPreTrainingLoss):
    _keys_to_ignore_on_load_unexpected = ['position_ids', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias']

    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name='bert')
        self.nsp = TFBertNSPHead(config, name='nsp___cls')
        self.mlm = TFBertMLMHead(config, input_embeddings=self.bert.embeddings, name='mlm___cls')

    def get_lm_head(self) -> keras.layers.Layer:
        return self.mlm.predictions

    def get_prefix_bias_name(self) -> str:
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.mlm.name + '/' + self.mlm.predictions.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, next_sentence_label: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[TFBertForPreTrainingOutput, Tuple[tf.Tensor]]:
        """
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        next_sentence_label (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Return:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFBertForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = TFBertForPreTraining.from_pretrained("google-bert/bert-base-uncased")
        >>> input_ids = tokenizer("Hello, my dog is cute", add_special_tokens=True, return_tensors="tf")
        >>> # Batch size 1

        >>> outputs = model(input_ids)
        >>> prediction_logits, seq_relationship_logits = outputs[:2]
        ```"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)
        seq_relationship_score = self.nsp(pooled_output=pooled_output)
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            d_labels = {'labels': labels}
            d_labels['next_sentence_label'] = next_sentence_label
            total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, seq_relationship_score))
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return TFBertForPreTrainingOutput(loss=total_loss, prediction_logits=prediction_scores, seq_relationship_logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'bert', None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        if getattr(self, 'nsp', None) is not None:
            with tf.name_scope(self.nsp.name):
                self.nsp.build(None)
        if getattr(self, 'mlm', None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)