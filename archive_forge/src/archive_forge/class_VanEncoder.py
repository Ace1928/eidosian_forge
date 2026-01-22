import math
from collections import OrderedDict
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....activations import ACT2FN
from ....modeling_outputs import (
from ....modeling_utils import PreTrainedModel
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_van import VanConfig
class VanEncoder(nn.Module):
    """
    VanEncoder, consisting of multiple stages.
    """

    def __init__(self, config: VanConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        patch_sizes = config.patch_sizes
        strides = config.strides
        hidden_sizes = config.hidden_sizes
        depths = config.depths
        mlp_ratios = config.mlp_ratios
        drop_path_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        for num_stage, (patch_size, stride, hidden_size, depth, mlp_expantion, drop_path_rate) in enumerate(zip(patch_sizes, strides, hidden_sizes, depths, mlp_ratios, drop_path_rates)):
            is_first_stage = num_stage == 0
            in_channels = hidden_sizes[num_stage - 1]
            if is_first_stage:
                in_channels = config.num_channels
            self.stages.append(VanStage(config, in_channels, hidden_size, patch_size=patch_size, stride=stride, depth=depth, mlp_ratio=mlp_expantion, drop_path_rate=drop_path_rate))

    def forward(self, hidden_state: torch.Tensor, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None
        for _, stage_module in enumerate(self.stages):
            hidden_state = stage_module(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=all_hidden_states)