from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
class TFLxmertEncoder(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.visn_fc = TFLxmertVisualFeatureEncoder(config, name='visn_fc')
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers
        self.layer = [TFLxmertLayer(config, name=f'layer_._{i}') for i in range(self.num_l_layers)]
        self.x_layers = [TFLxmertXLayer(config, name=f'x_layers_._{i}') for i in range(self.num_x_layers)]
        self.r_layers = [TFLxmertLayer(config, name=f'r_layers_._{i}') for i in range(self.num_r_layers)]
        self.config = config

    def call(self, lang_feats=None, lang_attention_mask=None, visual_feats=None, visual_pos=None, visual_attention_mask=None, output_attentions=None, training=False):
        vision_hidden_states = ()
        language_hidden_states = ()
        vision_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None
        visual_feats = self.visn_fc([visual_feats, visual_pos], training=training)
        for layer_module in self.layer:
            l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions, training=training)
            lang_feats = l_outputs[0]
            language_hidden_states = language_hidden_states + (lang_feats,)
            if language_attentions is not None:
                language_attentions = language_attentions + (l_outputs[1],)
        for layer_module in self.r_layers:
            v_outputs = layer_module(visual_feats, visual_attention_mask, output_attentions, training=training)
            visual_feats = v_outputs[0]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            if vision_attentions is not None:
                vision_attentions = vision_attentions + (v_outputs[1],)
        for layer_module in self.x_layers:
            x_outputs = layer_module(lang_feats, lang_attention_mask, visual_feats, visual_attention_mask, output_attentions, training=training)
            lang_feats, visual_feats = x_outputs[:2]
            vision_hidden_states = vision_hidden_states + (visual_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)
        visual_encoder_outputs = (vision_hidden_states, vision_attentions if output_attentions else None)
        lang_encoder_outputs = (language_hidden_states, language_attentions if output_attentions else None)
        return (visual_encoder_outputs, lang_encoder_outputs, cross_encoder_attentions if output_attentions else None)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'visn_fc', None) is not None:
            with tf.name_scope(self.visn_fc.name):
                self.visn_fc.build(None)
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
        if getattr(self, 'x_layers', None) is not None:
            for layer in self.x_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
        if getattr(self, 'r_layers', None) is not None:
            for layer in self.r_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)