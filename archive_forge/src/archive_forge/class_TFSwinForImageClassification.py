from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_swin import SwinConfig
@add_start_docstrings('\n    Swin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of\n    the [CLS] token) e.g. for ImageNet.\n    ', SWIN_START_DOCSTRING)
class TFSwinForImageClassification(TFSwinPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: SwinConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.swin = TFSwinMainLayer(config, name='swin')
        self.classifier = keras.layers.Dense(config.num_labels, name='classifier') if config.num_labels > 0 else keras.layers.Activation('linear', name='classifier')

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFSwinImageClassifierOutput, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple[tf.Tensor, ...], TFSwinImageClassifierOutput]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.swin(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output, training=training)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSwinImageClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, reshaped_hidden_states=outputs.reshaped_hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'swin', None) is not None:
            with tf.name_scope(self.swin.name):
                self.swin.build(None)
        if getattr(self, 'classifier', None) is not None:
            if hasattr(self.classifier, 'name'):
                with tf.name_scope(self.classifier.name):
                    self.classifier.build([None, None, self.swin.num_features])