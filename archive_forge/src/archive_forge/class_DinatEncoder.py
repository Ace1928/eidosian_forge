import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_dinat import DinatConfig
class DinatEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_levels = len(config.depths)
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.levels = nn.ModuleList([DinatStage(config=config, dim=int(config.embed_dim * 2 ** i_layer), depth=config.depths[i_layer], num_heads=config.num_heads[i_layer], dilations=config.dilations[i_layer], drop_path_rate=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])], downsample=DinatDownsampler if i_layer < self.num_levels - 1 else None) for i_layer in range(self.num_levels)])

    def forward(self, hidden_states: torch.Tensor, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, output_hidden_states_before_downsampling: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, DinatEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for i, layer_module in enumerate(self.levels):
            layer_outputs = layer_module(hidden_states, output_attentions)
            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            if output_hidden_states and output_hidden_states_before_downsampling:
                reshaped_hidden_state = hidden_states_before_downsampling.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and (not output_hidden_states_before_downsampling):
                reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if output_attentions:
                all_self_attentions += layer_outputs[2:]
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return DinatEncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions, reshaped_hidden_states=all_reshaped_hidden_states)