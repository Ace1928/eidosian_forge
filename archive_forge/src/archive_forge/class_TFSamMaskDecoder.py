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
class TFSamMaskDecoder(keras.layers.Layer):

    def __init__(self, config: SamMaskDecoderConfig, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1
        self.transformer = TFSamTwoWayTransformer(config, name='transformer')
        self.upscale_conv1 = keras.layers.Conv2DTranspose(self.hidden_size // 4, kernel_size=2, strides=2, name='upscale_conv1', data_format='channels_first')
        self.upscale_conv2 = keras.layers.Conv2DTranspose(self.hidden_size // 8, kernel_size=2, strides=2, name='upscale_conv2', data_format='channels_first')
        self.upscale_layer_norm = TFSamLayerNorm(self.hidden_size // 4, data_format='channels_first', name='upscale_layer_norm')
        self.activation = tf.nn.gelu
        mlps_list = []
        for i in range(self.num_mask_tokens):
            mlps_list += [TFSamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3, name=f'output_hypernetworks_mlps_._{i}')]
        self.output_hypernetworks_mlps = mlps_list
        self.iou_prediction_head = TFSamFeedForward(self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth, name='iou_prediction_head')

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        self.iou_token = self.add_weight(shape=(1, self.hidden_size), name='iou_token.weight', trainable=True)
        self.mask_tokens = self.add_weight(shape=(self.num_mask_tokens, self.hidden_size), name='mask_tokens.weight', trainable=True)
        if getattr(self, 'transformer', None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, 'upscale_conv1', None) is not None:
            with tf.name_scope(self.upscale_conv1.name):
                self.upscale_conv1.build([None, self.hidden_size, None, None])
        if getattr(self, 'upscale_conv2', None) is not None:
            with tf.name_scope(self.upscale_conv2.name):
                self.upscale_conv2.build([None, self.hidden_size // 4, None, None])
        if getattr(self, 'upscale_layer_norm', None) is not None:
            with tf.name_scope(self.upscale_layer_norm.name):
                self.upscale_layer_norm.build(None)
        if getattr(self, 'iou_prediction_head', None) is not None:
            with tf.name_scope(self.iou_prediction_head.name):
                self.iou_prediction_head.build(None)
        for mlp in self.output_hypernetworks_mlps:
            with tf.name_scope(mlp.name):
                mlp.build(None)

    def call(self, image_embeddings: tf.Tensor, image_positional_embeddings: tf.Tensor, sparse_prompt_embeddings: tf.Tensor, dense_prompt_embeddings: tf.Tensor, multimask_output: bool, output_attentions: Optional[bool]=None) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size, num_channels, height, width = shape_list(image_embeddings)
        point_batch_size = tf.math.maximum(1, tf.shape(sparse_prompt_embeddings)[1])
        output_tokens = tf.concat([self.iou_token, self.mask_tokens], axis=0)
        output_tokens = tf.tile(output_tokens[None, None, :], [batch_size, point_batch_size, 1, 1])
        if shape_list(sparse_prompt_embeddings)[1] != 0:
            tokens = tf.concat((output_tokens, sparse_prompt_embeddings), axis=2)
        else:
            tokens = output_tokens
        point_embeddings = tf.cast(tokens, self.iou_token.dtype)
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = tf.repeat(image_embeddings, point_batch_size, axis=0)
        image_positional_embeddings = tf.repeat(image_positional_embeddings, point_batch_size, axis=0)
        point_embedding, image_embeddings, attentions = self.transformer(point_embeddings=point_embeddings, image_embeddings=image_embeddings, image_positional_embeddings=image_positional_embeddings, output_attentions=output_attentions)
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1:1 + self.num_mask_tokens, :]
        image_embeddings = tf.transpose(image_embeddings, perm=(0, 1, 3, 2))
        image_embeddings = tf.reshape(image_embeddings, [batch_size * point_batch_size, num_channels, height, width])
        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = tf.stack(hyper_in_list, axis=2)
        _, num_channels, height, width = shape_list(upscaled_embedding)
        upscaled_embedding = tf.reshape(upscaled_embedding, [batch_size, point_batch_size, num_channels, height * width])
        masks = tf.reshape(hyper_in @ upscaled_embedding, [batch_size, point_batch_size, -1, height, width])
        iou_pred = self.iou_prediction_head(iou_token_out)
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]
        outputs = (masks, iou_pred)
        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)
        return outputs