from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_led import LEDConfig
@keras_serializable
class TFLEDMainLayer(keras.layers.Layer):
    config_class = LEDConfig

    def __init__(self, config: LEDConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.shared = keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.d_model, embeddings_initializer=keras.initializers.TruncatedNormal(stddev=self.config.init_std), name='led.shared')
        self.shared.load_weight_prefix = 'led.shared'
        self.encoder = TFLEDEncoder(config, self.shared, name='encoder')
        self.decoder = TFLEDDecoder(config, self.shared, name='decoder')

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, encoder_outputs: Optional[Union[Tuple, TFLEDEncoderBaseModelOutput]]=None, global_attention_mask=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False, **kwargs):
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            use_cache = False
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        elif return_dict and (not isinstance(encoder_outputs, TFLEDEncoderBaseModelOutput)):
            encoder_outputs = TFLEDEncoderBaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        elif not return_dict and (not isinstance(encoder_outputs, tuple)):
            encoder_outputs = encoder_outputs.to_tuple()
        decoder_outputs = self.decoder(decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, encoder_head_mask=head_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return TFLEDSeq2SeqModelOutput(last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions, encoder_global_attentions=encoder_outputs.global_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        with tf.name_scope(self.shared.load_weight_prefix + '/' + self.shared.name + '/'):
            self.shared.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'decoder', None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)