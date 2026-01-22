from typing import Mapping
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxSeq2SeqConfigWithPast
from ...utils import logging
class MT5OnnxConfig(OnnxSeq2SeqConfigWithPast):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = {'input_ids': {0: 'batch', 1: 'encoder_sequence'}, 'attention_mask': {0: 'batch', 1: 'encoder_sequence'}}
        if self.use_past:
            common_inputs['attention_mask'][1] = 'past_encoder_sequence + sequence'
            common_inputs['decoder_input_ids'] = {0: 'batch'}
            common_inputs['decoder_attention_mask'] = {0: 'batch', 1: 'past_decoder_sequence + sequence'}
        else:
            common_inputs['decoder_input_ids'] = {0: 'batch', 1: 'decoder_sequence'}
            common_inputs['decoder_attention_mask'] = {0: 'batch', 1: 'decoder_sequence'}
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction='inputs')
        return common_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13

    @property
    def atol_for_validation(self) -> float:
        return 0.0005