import collections
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_bit import BitConfig
class BitEncoder(nn.Module):

    def __init__(self, config: BitConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        prev_chs = config.embedding_size
        current_stride = 4
        dilation = 1
        layer_dropouts = [x.tolist() for x in torch.Tensor(np.linspace(0, config.drop_path_rate, sum(config.depths))).split(config.depths)]
        for stage_idx, (current_depth, current_hidden_size, layer_dropout) in enumerate(zip(config.depths, config.hidden_sizes, layer_dropouts)):
            out_channels, stride, dilation = self._get_updated_hyperparameters(stage_idx, current_stride, current_hidden_size, dilation, config)
            stage = BitStage(config, prev_chs, out_channels, stride=stride, dilation=dilation, depth=current_depth, layer_dropout=layer_dropout)
            prev_chs = out_channels
            current_stride *= stride
            self.stages.add_module(str(stage_idx), stage)

    def _get_updated_hyperparameters(self, stage_idx, current_stride, current_hidden_size, dilation, config):
        out_channels = make_div(current_hidden_size * config.width_factor)
        stride = 1 if stage_idx == 0 else 2
        if current_stride >= config.output_stride:
            dilation *= stride
            stride = 1
        return (out_channels, stride, dilation)

    def forward(self, hidden_state: Tensor, output_hidden_states: bool=False, return_dict: bool=True) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)
            hidden_state = stage_module(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)