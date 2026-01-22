from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig
@add_start_docstrings('\n    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', RESNET_START_DOCSTRING)
class TFResNetForImageClassification(TFResNetPreTrainedModel, TFSequenceClassificationLoss):

    def __init__(self, config: ResNetConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.resnet = TFResNetMainLayer(config, name='resnet')
        self.classifier_layer = keras.layers.Dense(config.num_labels, name='classifier.1') if config.num_labels > 0 else keras.layers.Activation('linear', name='classifier.1')
        self.config = config

    def classifier(self, x: tf.Tensor) -> tf.Tensor:
        x = keras.layers.Flatten()(x)
        logits = self.classifier_layer(x)
        return logits

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=TFImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    @unpack_inputs
    def call(self, pixel_values: tf.Tensor=None, labels: tf.Tensor=None, output_hidden_states: bool=None, return_dict: bool=None, training: bool=False) -> Union[Tuple[tf.Tensor], TFImageClassifierOutputWithNoAttention]:
        """
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'resnet', None) is not None:
            with tf.name_scope(self.resnet.name):
                self.resnet.build(None)
        if getattr(self, 'classifier_layer', None) is not None:
            with tf.name_scope(self.classifier_layer.name):
                self.classifier_layer.build([None, None, self.config.hidden_sizes[-1]])