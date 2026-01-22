import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
class CLIPOnnxConfig(OnnxConfig):

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'}), ('pixel_values', {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width'}), ('attention_mask', {0: 'batch', 1: 'sequence'})])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([('logits_per_image', {0: 'batch'}), ('logits_per_text', {0: 'batch'}), ('text_embeds', {0: 'batch'}), ('image_embeds', {0: 'batch'})])

    @property
    def atol_for_validation(self) -> float:
        return 0.0001

    def generate_dummy_inputs(self, processor: 'ProcessorMixin', batch_size: int=-1, seq_length: int=-1, framework: Optional['TensorType']=None) -> Mapping[str, Any]:
        text_input_dict = super().generate_dummy_inputs(processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework)
        image_input_dict = super().generate_dummy_inputs(processor.image_processor, batch_size=batch_size, framework=framework)
        return {**text_input_dict, **image_input_dict}

    @property
    def default_onnx_opset(self) -> int:
        return 14