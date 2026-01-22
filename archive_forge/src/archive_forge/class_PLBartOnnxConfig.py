from collections import OrderedDict
from typing import Mapping
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast
from ...utils import logging
class PLBartOnnxConfig(OnnxConfigWithPast):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'}), ('attention_mask', {0: 'batch', 1: 'sequence'})])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.use_past:
            return OrderedDict([('last_hidden_state', {0: 'batch', 1: 'sequence'}), ('past_keys', {0: 'batch', 2: 'sequence'}), ('encoder_last_hidden_state', {0: 'batch', 1: 'sequence'})])
        else:
            return OrderedDict([('last_hidden_state', {0: 'batch', 1: 'sequence'}), ('encoder_last_hidden_state', {0: 'batch', 1: 'sequence'})])