from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
@add_start_docstrings('The bare MobileViT model outputting raw hidden-states without any specific head on top.', MOBILEVIT_START_DOCSTRING)
class TFMobileViTModel(TFMobileViTPreTrainedModel):

    def __init__(self, config: MobileViTConfig, expand_output: bool=True, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.expand_output = expand_output
        self.mobilevit = TFMobileViTMainLayer(config, expand_output=expand_output, name='mobilevit')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def call(self, pixel_values: tf.Tensor | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPooling]:
        output = self.mobilevit(pixel_values, output_hidden_states, return_dict, training=training)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'mobilevit', None) is not None:
            with tf.name_scope(self.mobilevit.name):
                self.mobilevit.build(None)