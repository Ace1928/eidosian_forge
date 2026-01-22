import math
from typing import Dict, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mobilevit import MobileViTConfig
class MobileViTEncoder(nn.Module):

    def __init__(self, config: MobileViTConfig) -> None:
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
        layer_1 = MobileViTMobileNetLayer(config, in_channels=config.neck_hidden_sizes[0], out_channels=config.neck_hidden_sizes[1], stride=1, num_stages=1)
        self.layer.append(layer_1)
        layer_2 = MobileViTMobileNetLayer(config, in_channels=config.neck_hidden_sizes[1], out_channels=config.neck_hidden_sizes[2], stride=2, num_stages=3)
        self.layer.append(layer_2)
        layer_3 = MobileViTLayer(config, in_channels=config.neck_hidden_sizes[2], out_channels=config.neck_hidden_sizes[3], stride=2, hidden_size=config.hidden_sizes[0], num_stages=2)
        self.layer.append(layer_3)
        if dilate_layer_4:
            dilation *= 2
        layer_4 = MobileViTLayer(config, in_channels=config.neck_hidden_sizes[3], out_channels=config.neck_hidden_sizes[4], stride=2, hidden_size=config.hidden_sizes[1], num_stages=4, dilation=dilation)
        self.layer.append(layer_4)
        if dilate_layer_5:
            dilation *= 2
        layer_5 = MobileViTLayer(config, in_channels=config.neck_hidden_sizes[4], out_channels=config.neck_hidden_sizes[5], stride=2, hidden_size=config.hidden_sizes[2], num_stages=3, dilation=dilation)
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