import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig
class TvpFrameDownPadPrompter(nn.Module):
    """
    Pad frames extracted from videos only at the bottom.
    """

    def __init__(self, config):
        if config.visual_prompter_apply not in ('add', 'replace', 'remove'):
            raise ValueError('`visual_prompter_apply` must be in (add, replace, remove)')
        super().__init__()
        self.visual_prompt_size = config.visual_prompt_size
        self.frame_num = config.frame_num
        self.max_img_size = config.max_img_size
        self.visual_prompter_apply = config.visual_prompter_apply
        self.pad_down = nn.Parameter(torch.randn([1, config.frame_num, 3, config.visual_prompt_size, config.max_img_size]))

    def forward(self, pixel_values):
        if self.visual_prompter_apply != 'add':
            visual_prompt_mask = torch.ones([self.max_img_size, self.max_img_size], dtype=pixel_values.dtype, device=pixel_values.device)
            visual_prompt_mask[self.max_img_size - self.visual_prompt_size:self.max_img_size, :] = 0.0
            pixel_values *= visual_prompt_mask
        if self.visual_prompter_apply != 'remove':
            prompt = torch.zeros([pixel_values.shape[0], pixel_values.shape[1], 3, self.max_img_size, self.max_img_size], device=pixel_values.device)
            start_point = self.max_img_size - self.visual_prompt_size
            prompt[:, :, :, start_point:self.max_img_size, :] = self.pad_down
            pixel_values += prompt.to(pixel_values.dtype)
        return pixel_values