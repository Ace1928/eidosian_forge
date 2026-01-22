from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras, shape_list, unpack_inputs
from ...utils import (
from ..bert.modeling_tf_bert import TFBertMainLayer
from .configuration_dpr import DPRConfig
class TFDPRSpanPredictor(TFPreTrainedModel):
    base_model_prefix = 'encoder'

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.encoder = TFDPRSpanPredictorLayer(config)

    @unpack_inputs
    def call(self, input_ids: tf.Tensor=None, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=False, training: bool=False) -> Union[TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs