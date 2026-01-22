from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
@keras_serializable
class TFWav2Vec2MainLayer(keras.layers.Layer):
    config_class = Wav2Vec2Config

    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.feature_extractor = TFWav2Vec2FeatureEncoder(config, name='feature_extractor')
        self.feature_projection = TFWav2Vec2FeatureProjection(config, name='feature_projection')
        if config.do_stable_layer_norm:
            self.encoder = TFWav2Vec2EncoderStableLayerNorm(config, name='encoder')
        else:
            self.encoder = TFWav2Vec2Encoder(config, name='encoder')

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if self.config.mask_time_prob > 0.0 or self.config.mask_feature_prob > 0.0:
            self.masked_spec_embed = self.add_weight(shape=(self.config.hidden_size,), initializer='uniform', trainable=True, name='masked_spec_embed')
        if getattr(self, 'feature_extractor', None) is not None:
            with tf.name_scope(self.feature_extractor.name):
                self.feature_extractor.build(None)
        if getattr(self, 'feature_projection', None) is not None:
            with tf.name_scope(self.feature_projection.name):
                self.feature_projection.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths

    def _mask_hidden_states(self, hidden_states: tf.Tensor, mask_time_indices: tf.Tensor | None=None):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        batch_size, sequence_length, hidden_size = shape_list(hidden_states)
        if not getattr(self.config, 'apply_spec_augment', True):
            return hidden_states
        if mask_time_indices is not None:
            hidden_states = tf.where(tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool), self.masked_spec_embed[tf.newaxis, tf.newaxis, :], hidden_states)
        elif self.config.mask_time_prob > 0:
            mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=self.config.mask_time_prob, mask_length=self.config.mask_time_length, min_masks=2)
            hidden_states = tf.where(tf.cast(mask_time_indices[:, :, tf.newaxis], tf.bool), self.masked_spec_embed[tf.newaxis, tf.newaxis, :], hidden_states)
        if self.config.mask_feature_prob > 0:
            mask_feature_indices = _compute_mask_indices((batch_size, hidden_size), mask_prob=self.config.mask_feature_prob, mask_length=self.config.mask_feature_length)
            hidden_states = tf.where(mask_feature_indices[:, tf.newaxis, :], hidden_states, 0)
        return hidden_states

    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False, **kwargs: Any):
        extract_features = self.feature_extractor(tf.cast(input_values, tf.float32), training=training)
        if attention_mask is not None:
            output_lengths = self._get_feat_extract_output_lengths(tf.reduce_sum(attention_mask, -1))
            attention_mask = tf.sequence_mask(output_lengths, maxlen=shape_list(extract_features)[1], dtype=extract_features.dtype)
        hidden_states, extract_features = self.feature_projection(extract_features, training=training)
        mask_time_indices = kwargs.get('mask_time_indices', None)
        if training:
            hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]
        return TFWav2Vec2BaseModelOutput(last_hidden_state=hidden_states, extract_features=extract_features, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)