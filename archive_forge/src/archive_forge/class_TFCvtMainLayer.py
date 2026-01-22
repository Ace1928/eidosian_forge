from __future__ import annotations
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_cvt import CvtConfig
@keras_serializable
class TFCvtMainLayer(keras.layers.Layer):
    """Construct the Cvt model."""
    config_class = CvtConfig

    def __init__(self, config: CvtConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.encoder = TFCvtEncoder(config, name='encoder')

    @unpack_inputs
    def call(self, pixel_values: TFModelInputType | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        encoder_outputs = self.encoder(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return TFBaseModelOutputWithCLSToken(last_hidden_state=sequence_output, cls_token_value=encoder_outputs.cls_token_value, hidden_states=encoder_outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)