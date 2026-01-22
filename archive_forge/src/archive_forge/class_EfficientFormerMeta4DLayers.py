import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class EfficientFormerMeta4DLayers(nn.Module):

    def __init__(self, config: EfficientFormerConfig, stage_idx: int):
        super().__init__()
        num_layers = config.depths[stage_idx] if stage_idx != -1 else config.depths[stage_idx] - config.num_meta3d_blocks
        drop_paths = [config.drop_path_rate * (block_idx + sum(config.depths[:stage_idx])) for block_idx in range(num_layers)]
        self.blocks = nn.ModuleList([EfficientFormerMeta4D(config, config.hidden_sizes[stage_idx], drop_path=drop_path) for drop_path in drop_paths])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
        return hidden_states