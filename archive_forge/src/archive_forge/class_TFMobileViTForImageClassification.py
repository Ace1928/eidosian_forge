from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
@add_start_docstrings('\n    MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', MOBILEVIT_START_DOCSTRING)
class TFMobileViTForImageClassification(TFMobileViTPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: MobileViTConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.mobilevit = TFMobileViTMainLayer(config, name='mobilevit')
        self.dropout = keras.layers.Dropout(config.classifier_dropout_prob)
        self.classifier = keras.layers.Dense(config.num_labels, name='classifier') if config.num_labels > 0 else tf.identity
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def call(self, pixel_values: tf.Tensor | None=None, output_hidden_states: Optional[bool]=None, labels: tf.Tensor | None=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[tuple, TFImageClassifierOutputWithNoAttention]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mobilevit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(self.dropout(pooled_output, training=training))
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'mobilevit', None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)
        if getattr(self, 'classifier', None) is not None:
            if hasattr(self.classifier, 'name'):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.config.neck_hidden_sizes[-1]])