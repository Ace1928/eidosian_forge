from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
class VisionEncoderDecoderEncoderOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse('1.11')

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([('pixel_values', {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width'})])

    @property
    def atol_for_validation(self) -> float:
        return 0.0001

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict({'last_hidden_state': {0: 'batch', 1: 'encoder_sequence'}})