from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
@keras_serializable
class TFWhisperEncoder(keras.layers.Layer):
    config_class = WhisperConfig
    '\n    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a\n    [`TFWhisperEncoderLayer`].\n\n    Args:\n        config: WhisperConfig\n        embed_tokens (TFWhisperEmbedding): output embedding\n    '

    def __init__(self, config: WhisperConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layerdrop = config.encoder_layerdrop
        self.embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0
        self.conv1 = keras.layers.Conv1D(self.embed_dim, kernel_size=3, strides=1, padding='valid', name='conv1')
        self.conv2 = keras.layers.Conv1D(self.embed_dim, kernel_size=3, strides=2, padding='valid', name='conv2')
        self.embed_positions = TFWhisperPositionalEmbedding(num_positions=self.max_source_positions, embedding_dim=self.embed_dim, embedding_initializer=sinusoidal_embedding_init, name='embed_positions')
        self.embed_positions.trainable = False
        self.encoder_layers = [TFWhisperEncoderLayer(config, name=f'layers.{i}') for i in range(config.encoder_layers)]
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm')
        self.dropout = keras.layers.Dropout(config.dropout)

    @unpack_inputs
    def call(self, input_features=None, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        """
        Args:
            input_features (`tf.Tensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of fbank features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the fbank features,
                padding and conversion into a tensor of type `tf.Tensor`. See [`~WhisperFeatureExtractor.__call__`]
            head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_features = tf.transpose(input_features, perm=(0, 2, 1))
        input_features = tf.pad(input_features, [[0, 0], [1, 1], [0, 0]])
        inputs_embeds = keras.activations.gelu(self.conv1(input_features))
        inputs_embeds = tf.pad(inputs_embeds, [[0, 0], [1, 1], [0, 0]])
        inputs_embeds = keras.activations.gelu(self.conv2(inputs_embeds))
        inputs_embeds = tf.transpose(inputs_embeds, perm=(0, 1, 2))
        embed_pos = self.embed_positions(input_ids=tf.zeros((1, self.max_source_positions), dtype=tf.int32))
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.dropout(hidden_states, training=training)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            tf.debugging.assert_equal(shape_list(head_mask)[0], len(self.encoder_layers), message=f'The head_mask should be specified for {len(self.encoder_layers)} layers, but it is for {shape_list(head_mask)[0]}.')
        for idx, encoder_layer in enumerate(self.encoder_layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if training and dropout_probability < self.layerdrop:
                continue
            hidden_states, attn = encoder_layer(hidden_states, None, layer_head_mask=head_mask[idx] if head_mask is not None else None, training=training)
            if output_attentions:
                all_attentions += (attn,)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv1', None) is not None:
            with tf.name_scope(self.conv1.name):
                self.conv1.build([None, None, self.num_mel_bins])
        if getattr(self, 'conv2', None) is not None:
            with tf.name_scope(self.conv2.name):
                self.conv2.build([None, None, self.embed_dim])
        if getattr(self, 'embed_positions', None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        if getattr(self, 'encoder_layers', None) is not None:
            for layer in self.encoder_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)