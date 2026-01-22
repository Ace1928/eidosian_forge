from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilevitv2 import MobileViTV2Config
class MobileViTV2Encoder(nn.Module):

    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList()
        self.gradient_checkpointing = False
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True
        dilation = 1
        layer_0_dim = make_divisible(clip(value=32 * config.width_multiplier, min_val=16, max_val=64), divisor=8, min_value=16)
        layer_1_dim = make_divisible(64 * config.width_multiplier, divisor=16)
        layer_2_dim = make_divisible(128 * config.width_multiplier, divisor=8)
        layer_3_dim = make_divisible(256 * config.width_multiplier, divisor=8)
        layer_4_dim = make_divisible(384 * config.width_multiplier, divisor=8)
        layer_5_dim = make_divisible(512 * config.width_multiplier, divisor=8)
        layer_1 = MobileViTV2MobileNetLayer(config, in_channels=layer_0_dim, out_channels=layer_1_dim, stride=1, num_stages=1)
        self.layer.append(layer_1)
        layer_2 = MobileViTV2MobileNetLayer(config, in_channels=layer_1_dim, out_channels=layer_2_dim, stride=2, num_stages=2)
        self.layer.append(layer_2)
        layer_3 = MobileViTV2Layer(config, in_channels=layer_2_dim, out_channels=layer_3_dim, attn_unit_dim=make_divisible(config.base_attn_unit_dims[0] * config.width_multiplier, divisor=8), n_attn_blocks=config.n_attn_blocks[0])
        self.layer.append(layer_3)
        if dilate_layer_4:
            dilation *= 2
        layer_4 = MobileViTV2Layer(config, in_channels=layer_3_dim, out_channels=layer_4_dim, attn_unit_dim=make_divisible(config.base_attn_unit_dims[1] * config.width_multiplier, divisor=8), n_attn_blocks=config.n_attn_blocks[1], dilation=dilation)
        self.layer.append(layer_4)
        if dilate_layer_5:
            dilation *= 2
        layer_5 = MobileViTV2Layer(config, in_channels=layer_4_dim, out_channels=layer_5_dim, attn_unit_dim=make_divisible(config.base_attn_unit_dims[2] * config.width_multiplier, divisor=8), n_attn_blocks=config.n_attn_blocks[2], dilation=dilation)
        self.layer.append(layer_5)

    def forward(self, hidden_states: torch.Tensor, output_hidden_states: bool=False, return_dict: bool=True) -> Union[tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(layer_module.__call__, hidden_states)
            else:
                hidden_states = layer_module(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)