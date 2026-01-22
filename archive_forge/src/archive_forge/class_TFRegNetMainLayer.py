from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
@keras_serializable
class TFRegNetMainLayer(keras.layers.Layer):
    config_class = RegNetConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedder = TFRegNetEmbeddings(config, name='embedder')
        self.encoder = TFRegNetEncoder(config, name='encoder')
        self.pooler = keras.layers.GlobalAveragePooling2D(keepdims=True, name='pooler')

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> TFBaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embedding_output = self.embedder(pixel_values, training=training)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        pooled_output = tf.transpose(pooled_output, perm=(0, 3, 1, 2))
        last_hidden_state = tf.transpose(last_hidden_state, perm=(0, 3, 1, 2))
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPoolingAndNoAttention(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embedder', None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'pooler', None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build((None, None, None, None))