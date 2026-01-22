from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
@add_start_docstrings('\n    Data2VecVision Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.\n    ', DATA2VEC_VISION_START_DOCSTRING)
class TFData2VecVisionForSemanticSegmentation(TFData2VecVisionPreTrainedModel):

    def __init__(self, config: Data2VecVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.data2vec_vision = TFData2VecVisionMainLayer(config, add_pooling_layer=False, name='data2vec_vision')
        self.fpn1 = [keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name='fpn1.0'), keras.layers.BatchNormalization(name='fpn1.1', momentum=0.9, epsilon=1e-05), keras.layers.Activation('gelu'), keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name='fpn1.3')]
        self.fpn2 = [keras.layers.Conv2DTranspose(config.hidden_size, kernel_size=2, strides=2, name='fpn2.0')]
        self.fpn3 = tf.identity
        self.fpn4 = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.decode_head = TFData2VecVisionUperHead(config, name='decode_head')
        self.auxiliary_head = TFData2VecVisionFCNHead(config, name='auxiliary_head') if config.use_auxiliary_head else None

    def compute_loss(self, logits, auxiliary_logits, labels):
        if len(shape_list(labels)) > 3:
            label_interp_shape = shape_list(labels)[1:-1]
        else:
            label_interp_shape = shape_list(labels)[-2:]
        upsampled_logits = tf.image.resize(logits, size=label_interp_shape, method='bilinear')
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = tf.image.resize(auxiliary_logits, size=label_interp_shape, method='bilinear')
        loss_fct = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        def masked_loss(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, self.config.semantic_loss_ignore_index))
            loss_ = loss_fct(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            reduced_masked_loss = tf.reduce_sum(loss_) / tf.reduce_sum(mask)
            return tf.reshape(reduced_masked_loss, (1,))
        main_loss = masked_loss(labels, upsampled_logits)
        auxiliary_loss = masked_loss(labels, upsampled_auxiliary_logits)
        loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss
        return loss

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, pixel_values: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, labels: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, TFSemanticSegmenterOutput]:
        """
        labels (`tf.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFData2VecVisionForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
        >>> model = TFData2VecVisionForSemanticSegmentation.from_pretrained("facebook/data2vec-vision-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        outputs = self.data2vec_vision(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict)
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        patch_resolution = self.config.image_size // self.config.patch_size

        def reshape_features(x):
            x = tf.reshape(x, (-1, patch_resolution, patch_resolution, self.config.hidden_size))
            return x
        features = [reshape_features(x[:, 1:, :]) for x in features]
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for module in ops[0]:
            features[0] = module(features[0])
        features[1] = ops[1][0](features[1])
        for i in range(len(features[2:])):
            features[i + 2] = ops[i + 2](features[i + 2])
        logits = self.decode_head(features)
        transposed_logits = tf.transpose(logits, perm=[0, 3, 1, 2])
        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError('The number of labels should be greater than one')
            else:
                loss = self.compute_loss(logits, auxiliary_logits, labels)
        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        return TFSemanticSegmenterOutput(loss=loss, logits=transposed_logits, hidden_states=outputs.hidden_states if output_hidden_states else None, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'data2vec_vision', None) is not None:
            with tf.name_scope(self.data2vec_vision.name):
                self.data2vec_vision.build(None)
        if getattr(self, 'decode_head', None) is not None:
            with tf.name_scope(self.decode_head.name):
                self.decode_head.build(None)
        if getattr(self, 'auxiliary_head', None) is not None:
            with tf.name_scope(self.auxiliary_head.name):
                self.auxiliary_head.build(None)
        if getattr(self, 'fpn1', None) is not None:
            with tf.name_scope(self.fpn1[0].name):
                self.fpn1[0].build([None, None, None, self.config.hidden_size])
            with tf.name_scope(self.fpn1[1].name):
                self.fpn1[1].build((None, None, None, self.config.hidden_size))
            with tf.name_scope(self.fpn1[3].name):
                self.fpn1[3].build([None, None, None, self.config.hidden_size])
        if getattr(self, 'fpn2', None) is not None:
            with tf.name_scope(self.fpn2[0].name):
                self.fpn2[0].build([None, None, None, self.config.hidden_size])