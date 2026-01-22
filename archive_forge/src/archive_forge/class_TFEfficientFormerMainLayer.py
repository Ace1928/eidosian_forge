import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
@keras_serializable
class TFEfficientFormerMainLayer(keras.layers.Layer):
    config_class = EfficientFormerConfig

    def __init__(self, config: EfficientFormerConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.patch_embed = TFEfficientFormerConvStem(config, config.hidden_sizes[0], name='patch_embed')
        self.encoder = TFEfficientFormerEncoder(config, name='encoder')
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')

    @unpack_inputs
    def call(self, pixel_values: Optional[tf.Tensor]=None, output_attentions: Optional[tf.Tensor]=None, output_hidden_states: Optional[tf.Tensor]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        embedding_output = self.patch_embed(pixel_values, training=training)
        encoder_outputs = self.encoder(hidden_states=embedding_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output, training=training)
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1][:-1]]) + (encoder_outputs[1][-1],)
        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return TFBaseModelOutput(last_hidden_state=sequence_output, hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embed', None) is not None:
            with tf.name_scope(self.patch_embed.name):
                self.patch_embed.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_sizes[-1]])