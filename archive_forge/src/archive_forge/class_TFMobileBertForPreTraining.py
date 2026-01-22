from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
@add_start_docstrings('\n    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a\n    `next sentence prediction (classification)` head.\n    ', MOBILEBERT_START_DOCSTRING)
class TFMobileBertForPreTraining(TFMobileBertPreTrainedModel, TFMobileBertPreTrainingLoss):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.mobilebert = TFMobileBertMainLayer(config, name='mobilebert')
        self.predictions = TFMobileBertMLMHead(config, name='predictions___cls')
        self.seq_relationship = TFMobileBertOnlyNSPHead(config, name='seq_relationship___cls')

    def get_lm_head(self):
        return self.predictions.predictions

    def get_prefix_bias_name(self):
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.predictions.name + '/' + self.predictions.predictions.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFMobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: np.ndarray | tf.Tensor | None=None, next_sentence_label: np.ndarray | tf.Tensor | None=None, training: Optional[bool]=False) -> Union[Tuple, TFMobileBertForPreTrainingOutput]:
        """
        Return:

        Examples:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, TFMobileBertForPreTraining

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = TFMobileBertForPreTraining.from_pretrained("google/mobilebert-uncased")
        >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
        >>> outputs = model(input_ids)
        >>> prediction_scores, seq_relationship_scores = outputs[:2]
        ```"""
        outputs = self.mobilebert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        total_loss = None
        if labels is not None and next_sentence_label is not None:
            d_labels = {'labels': labels}
            d_labels['next_sentence_label'] = next_sentence_label
            total_loss = self.hf_compute_loss(labels=d_labels, logits=(prediction_scores, seq_relationship_score))
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return TFMobileBertForPreTrainingOutput(loss=total_loss, prediction_logits=prediction_scores, seq_relationship_logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'mobilebert', None) is not None:
            with tf.name_scope(self.mobilebert.name):
                self.mobilebert.build(None)
        if getattr(self, 'predictions', None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
        if getattr(self, 'seq_relationship', None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                self.seq_relationship.build(None)

    def tf_to_pt_weight_rename(self, tf_weight):
        if tf_weight == 'cls.predictions.decoder.weight':
            return (tf_weight, 'mobilebert.embeddings.word_embeddings.weight')
        else:
            return (tf_weight,)