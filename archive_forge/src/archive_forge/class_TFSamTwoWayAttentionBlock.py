from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
class TFSamTwoWayAttentionBlock(keras.layers.Layer):

    def __init__(self, config, attention_downsample_rate: int=2, skip_first_layer_pe: bool=False, **kwargs):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optionalk*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps
        self.self_attn = TFSamAttention(config, downsample_rate=1, name='self_attn')
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name='layer_norm1')
        self.cross_attn_token_to_image = TFSamAttention(config, downsample_rate=attention_downsample_rate, name='cross_attn_token_to_image')
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name='layer_norm2')
        self.mlp = TFSamMLPBlock(config, name='mlp')
        self.layer_norm3 = keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name='layer_norm3')
        self.layer_norm4 = keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name='layer_norm4')
        self.cross_attn_image_to_token = TFSamAttention(config, downsample_rate=attention_downsample_rate, name='cross_attn_image_to_token')
        self.skip_first_layer_pe = skip_first_layer_pe

    def call(self, queries: tf.Tensor, keys: tf.Tensor, query_point_embedding: tf.Tensor, key_point_embedding: tf.Tensor, output_attentions: bool=False):
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_token_to_image(query=query, key=key, value=keys)
        queries = queries + attn_out
        queries = self.layer_norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out
        keys = self.layer_norm4(keys)
        outputs = (queries, keys)
        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self_attn', None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        if getattr(self, 'layer_norm1', None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, None, self.hidden_size])
        if getattr(self, 'cross_attn_token_to_image', None) is not None:
            with tf.name_scope(self.cross_attn_token_to_image.name):
                self.cross_attn_token_to_image.build(None)
        if getattr(self, 'layer_norm2', None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, None, self.hidden_size])
        if getattr(self, 'mlp', None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, 'layer_norm3', None) is not None:
            with tf.name_scope(self.layer_norm3.name):
                self.layer_norm3.build([None, None, None, self.hidden_size])
        if getattr(self, 'layer_norm4', None) is not None:
            with tf.name_scope(self.layer_norm4.name):
                self.layer_norm4.build([None, None, None, self.hidden_size])
        if getattr(self, 'cross_attn_image_to_token', None) is not None:
            with tf.name_scope(self.cross_attn_image_to_token.name):
                self.cross_attn_image_to_token.build(None)