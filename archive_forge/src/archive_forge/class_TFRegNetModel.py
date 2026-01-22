from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
@add_start_docstrings('The bare RegNet model outputting raw features without any specific head on top.', REGNET_START_DOCSTRING)
class TFRegNetModel(TFRegNetPreTrainedModel):

    def __init__(self, config: RegNetConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.regnet = TFRegNetMainLayer(config, name='regnet')

    @unpack_inputs
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFBaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def call(self, pixel_values: tf.Tensor, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPoolingAndNoAttention, Tuple[tf.Tensor]]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.regnet(pixel_values=pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if not return_dict:
            return (outputs[0],) + outputs[1:]
        return TFBaseModelOutputWithPoolingAndNoAttention(last_hidden_state=outputs.last_hidden_state, pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'regnet', None) is not None:
            with tf.name_scope(self.regnet.name):
                self.regnet.build(None)