from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_albert import AlbertConfig
class TFAlbertTransformer(keras.layers.Layer):

    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        self.layers_per_group = int(config.num_hidden_layers / config.num_hidden_groups)
        self.embedding_hidden_mapping_in = keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='embedding_hidden_mapping_in')
        self.albert_layer_groups = [TFAlbertLayerGroup(config, name=f'albert_layer_groups_._{i}') for i in range(config.num_hidden_groups)]
        self.config = config

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        hidden_states = self.embedding_hidden_mapping_in(inputs=hidden_states)
        all_attentions = () if output_attentions else None
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for i in range(self.num_hidden_layers):
            group_idx = int(i / (self.num_hidden_layers / self.num_hidden_groups))
            layer_group_output = self.albert_layer_groups[group_idx](hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask[group_idx * self.layers_per_group:(group_idx + 1) * self.layers_per_group], output_attentions=output_attentions, output_hidden_states=output_hidden_states, training=training)
            hidden_states = layer_group_output[0]
            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embedding_hidden_mapping_in', None) is not None:
            with tf.name_scope(self.embedding_hidden_mapping_in.name):
                self.embedding_hidden_mapping_in.build([None, None, self.config.embedding_size])
        if getattr(self, 'albert_layer_groups', None) is not None:
            for layer in self.albert_layer_groups:
                with tf.name_scope(layer.name):
                    layer.build(None)