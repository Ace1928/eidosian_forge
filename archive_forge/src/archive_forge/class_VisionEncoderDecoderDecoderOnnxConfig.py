from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
class VisionEncoderDecoderDecoderOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict()
        common_inputs['input_ids'] = {0: 'batch', 1: 'past_decoder_sequence + sequence'}
        common_inputs['attention_mask'] = {0: 'batch', 1: 'past_decoder_sequence + sequence'}
        common_inputs['encoder_hidden_states'] = {0: 'batch', 1: 'encoder_sequence'}
        return common_inputs

    def generate_dummy_inputs(self, tokenizer: 'PreTrainedTokenizerBase', batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional['TensorType']=None) -> Mapping[str, Any]:
        import torch
        common_inputs = OrderedDict()
        dummy_input = super().generate_dummy_inputs(tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework)
        batch, encoder_sequence = dummy_input['input_ids'].shape
        encoder_hidden_states_shape = (batch, encoder_sequence, self._config.encoder_hidden_size)
        common_inputs['input_ids'] = dummy_input.pop('input_ids')
        common_inputs['attention_mask'] = dummy_input.pop('attention_mask')
        common_inputs['encoder_hidden_states'] = torch.zeros(encoder_hidden_states_shape)
        return common_inputs