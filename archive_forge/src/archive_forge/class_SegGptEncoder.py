import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seggpt import SegGptConfig
from ..deprecated._archive_maps import SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class SegGptEncoder(nn.Module):

    def __init__(self, config: SegGptConfig) -> None:
        super().__init__()
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([SegGptLayer(config, dpr[i]) for i in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, feature_ensemble: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True) -> Union[tuple, SegGptEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        intermediate_hidden_states = []
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            ensemble_cond = 2 if self.config.merge_index > i else 1
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, ensemble_cond, feature_ensemble, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, ensemble_cond, feature_ensemble, output_attentions)
            hidden_states = layer_outputs[0]
            if i == self.config.merge_index:
                hidden_states = (hidden_states[:hidden_states.shape[0] // 2] + hidden_states[hidden_states.shape[0] // 2:]) * 0.5
            if i in self.config.intermediate_hidden_state_indices:
                intermediate_hidden_states.append(self.layernorm(hidden_states))
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions, intermediate_hidden_states] if v is not None))
        return SegGptEncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions, intermediate_hidden_states=intermediate_hidden_states)