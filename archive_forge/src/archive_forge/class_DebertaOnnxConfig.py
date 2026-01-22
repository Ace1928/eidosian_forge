from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
class DebertaOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == 'multiple-choice':
            dynamic_axis = {0: 'batch', 1: 'choice', 2: 'sequence'}
        else:
            dynamic_axis = {0: 'batch', 1: 'sequence'}
        if self._config.type_vocab_size > 0:
            return OrderedDict([('input_ids', dynamic_axis), ('attention_mask', dynamic_axis), ('token_type_ids', dynamic_axis)])
        else:
            return OrderedDict([('input_ids', dynamic_axis), ('attention_mask', dynamic_axis)])

    @property
    def default_onnx_opset(self) -> int:
        return 12

    def generate_dummy_inputs(self, preprocessor: Union['PreTrainedTokenizerBase', 'FeatureExtractionMixin'], batch_size: int=-1, seq_length: int=-1, num_choices: int=-1, is_pair: bool=False, framework: Optional['TensorType']=None, num_channels: int=3, image_width: int=40, image_height: int=40, tokenizer: 'PreTrainedTokenizerBase'=None) -> Mapping[str, Any]:
        dummy_inputs = super().generate_dummy_inputs(preprocessor=preprocessor, framework=framework)
        if self._config.type_vocab_size == 0 and 'token_type_ids' in dummy_inputs:
            del dummy_inputs['token_type_ids']
        return dummy_inputs