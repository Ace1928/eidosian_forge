from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras, shape_list, unpack_inputs
from ...utils import (
from ..bert.modeling_tf_bert import TFBertMainLayer
from .configuration_dpr import DPRConfig
@add_start_docstrings('The bare DPRContextEncoder transformer outputting pooler outputs as context representations.', TF_DPR_START_DOCSTRING)
class TFDPRContextEncoder(TFDPRPretrainedContextEncoder):

    def __init__(self, config: DPRConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.ctx_encoder = TFDPREncoderLayer(config, name='ctx_encoder')

    def get_input_embeddings(self):
        try:
            return self.ctx_encoder.bert_model.get_input_embeddings()
        except AttributeError:
            self.build()
            return self.ctx_encoder.bert_model.get_input_embeddings()

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TF_DPR_ENCODERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFDPRContextEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, training: bool=False) -> TFDPRContextEncoderOutput | Tuple[tf.Tensor, ...]:
        """
        Return:

        Examples:

        ```python
        >>> from transformers import TFDPRContextEncoder, DPRContextEncoderTokenizer

        >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        >>> model = TFDPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", from_pt=True)
        >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="tf")["input_ids"]
        >>> embeddings = model(input_ids).pooler_output
        ```
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = tf.ones(input_shape, dtype=tf.dtypes.int32) if input_ids is None else input_ids != self.config.pad_token_id
        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=tf.dtypes.int32)
        outputs = self.ctx_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if not return_dict:
            return outputs[1:]
        return TFDPRContextEncoderOutput(pooler_output=outputs.pooler_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'ctx_encoder', None) is not None:
            with tf.name_scope(self.ctx_encoder.name):
                self.ctx_encoder.build(None)