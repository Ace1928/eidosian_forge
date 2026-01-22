from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
@keras_serializable
class TFCLIPMainLayer(keras.layers.Layer):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(f'config.text_config is expected to be of type CLIPTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type CLIPVisionConfig but is of type {type(config.vision_config)}.')
        self.config = config
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.text_model = TFCLIPTextTransformer(text_config, name='text_model')
        self.vision_model = TFCLIPVisionTransformer(vision_config, name='vision_model')
        self.visual_projection = keras.layers.Dense(units=self.projection_dim, kernel_initializer=get_initializer(vision_config.hidden_size ** (-0.5) * self.config.initializer_factor), use_bias=False, name='visual_projection')
        self.text_projection = keras.layers.Dense(units=self.projection_dim, kernel_initializer=get_initializer(text_config.hidden_size ** (-0.5) * self.config.initializer_factor), use_bias=False, name='text_projection')
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

    def build(self, input_shape: tf.TensorShape=None):
        self.logit_scale = self.add_weight(shape=(1,), initializer=keras.initializers.Constant(self.config.logit_scale_init_value), trainable=True, name='logit_scale')
        if self.built:
            return
        self.built = True
        if getattr(self, 'text_model', None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
        if getattr(self, 'vision_model', None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        if getattr(self, 'visual_projection', None) is not None:
            with tf.name_scope(self.visual_projection.name):
                self.visual_projection.build([None, None, self.vision_embed_dim])
        if getattr(self, 'text_projection', None) is not None:
            with tf.name_scope(self.text_projection.name):
                self.text_projection.build([None, None, self.text_embed_dim])

    @unpack_inputs
    def get_text_features(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if input_ids is None:
            raise ValueError('You have to specify either input_ids')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = text_outputs[1]
        text_features = self.text_projection(inputs=pooled_output)
        return text_features

    @unpack_inputs
    def get_image_features(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = vision_outputs[1]
        image_features = self.visual_projection(inputs=pooled_output)
        return image_features

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, pixel_values: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFCLIPOutput, Tuple[tf.Tensor]]:
        if input_ids is None:
            raise ValueError('You have to specify either input_ids')
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(inputs=image_embeds)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(inputs=text_embeds)
        image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord='euclidean', axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord='euclidean', axis=-1, keepdims=True)
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)
        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)
            loss = tf.reshape(loss, (1,))
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return TFCLIPOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)