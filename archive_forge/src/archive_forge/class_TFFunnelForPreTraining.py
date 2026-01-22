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
from .configuration_funnel import FunnelConfig
@add_start_docstrings('\n    Funnel model with a binary classification head on top as used during pretraining for identifying generated tokens.\n    ', FUNNEL_START_DOCSTRING)
class TFFunnelForPreTraining(TFFunnelPreTrainedModel):

    def __init__(self, config: FunnelConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.funnel = TFFunnelMainLayer(config, name='funnel')
        self.discriminator_predictions = TFFunnelDiscriminatorPredictions(config, name='discriminator_predictions')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFFunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False, **kwargs) -> Union[Tuple[tf.Tensor], TFFunnelForPreTrainingOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFFunnelForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/small")
        >>> model = TFFunnelForPreTraining.from_pretrained("funnel-transformer/small")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> logits = model(inputs).logits
        ```"""
        discriminator_hidden_states = self.funnel(input_ids, attention_mask, token_type_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        if not return_dict:
            return (logits,) + discriminator_hidden_states[1:]
        return TFFunnelForPreTrainingOutput(logits=logits, hidden_states=discriminator_hidden_states.hidden_states, attentions=discriminator_hidden_states.attentions)

    def serving_output(self, output):
        return TFFunnelForPreTrainingOutput(logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'funnel', None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
        if getattr(self, 'discriminator_predictions', None) is not None:
            with tf.name_scope(self.discriminator_predictions.name):
                self.discriminator_predictions.build(None)