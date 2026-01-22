from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilenet_v2 import MobileNetV2Config
@add_start_docstrings('The bare MobileNetV2 model outputting raw hidden-states without any specific head on top.', MOBILENET_V2_START_DOCSTRING)
class MobileNetV2Model(MobileNetV2PreTrainedModel):

    def __init__(self, config: MobileNetV2Config, add_pooling_layer: bool=True):
        super().__init__(config)
        self.config = config
        channels = [16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320]
        channels = [apply_depth_multiplier(config, x) for x in channels]
        strides = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
        self.conv_stem = MobileNetV2Stem(config, in_channels=config.num_channels, expanded_channels=apply_depth_multiplier(config, 32), out_channels=channels[0])
        current_stride = 2
        dilation = 1
        self.layer = nn.ModuleList()
        for i in range(16):
            if current_stride == config.output_stride:
                layer_stride = 1
                layer_dilation = dilation
                dilation *= strides[i]
            else:
                layer_stride = strides[i]
                layer_dilation = 1
                current_stride *= layer_stride
            self.layer.append(MobileNetV2InvertedResidual(config, in_channels=channels[i], out_channels=channels[i + 1], stride=layer_stride, dilation=layer_dilation))
        if config.finegrained_output and config.depth_multiplier < 1.0:
            output_channels = 1280
        else:
            output_channels = apply_depth_multiplier(config, 1280)
        self.conv_1x1 = MobileNetV2ConvLayer(config, in_channels=channels[-1], out_channels=output_channels, kernel_size=1)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.Tensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        hidden_states = self.conv_stem(pixel_values)
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        last_hidden_state = self.conv_1x1(hidden_states)
        if self.pooler is not None:
            pooled_output = torch.flatten(self.pooler(last_hidden_state), start_dim=1)
        else:
            pooled_output = None
        if not return_dict:
            return tuple((v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None))
        return BaseModelOutputWithPoolingAndNoAttention(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=all_hidden_states)