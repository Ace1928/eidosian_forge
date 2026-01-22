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
class TvpFramePadPrompter(nn.Module):
    """
    Pad frames extracted from videos in the surroundings.
    """

    def __init__(self, config):
        if config.visual_prompter_apply not in ('add', 'replace', 'remove'):
            raise ValueError('`visual_prompter_apply` must be in (add, replace, remove)')
        super().__init__()
        self.num_frames = config.num_frames
        self.max_img_size = config.max_img_size
        self.visual_prompter_apply = config.visual_prompter_apply
        self.base_size = config.max_img_size - config.visual_prompt_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, config.num_frames, 3, config.visual_prompt_size, config.max_img_size]))
        self.pad_down = nn.Parameter(torch.randn([1, config.num_frames, 3, config.visual_prompt_size, config.max_img_size]))
        self.pad_left = nn.Parameter(torch.randn([1, config.num_frames, 3, config.max_img_size - config.visual_prompt_size * 2, config.visual_prompt_size]))
        self.pad_right = nn.Parameter(torch.randn([1, config.num_frames, 3, config.max_img_size - config.visual_prompt_size * 2, config.visual_prompt_size]))

    def forward(self, pixel_values):
        if self.visual_prompter_apply not in ('add', 'remove', 'replace'):
            raise ValueError(f'Invalid visual_prompter_apply value {self.visual_prompter_apply}')
        if self.visual_prompter_apply in ('replace', 'remove'):
            visual_prompt_mask = torch.ones([self.max_img_size, self.max_img_size], dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values *= visual_prompt_mask
        if self.visual_prompter_apply in ('replace', 'add'):
            base = torch.zeros(1, self.num_frames, 3, self.base_size, self.base_size, device=pixel_values.device)
            prompt = torch.cat([self.pad_left, base, self.pad_right], dim=4)
            prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=3)
            prompt = torch.cat(pixel_values.size(0) * [prompt])
            pixel_values += prompt.to(pixel_values.dtype)
        return pixel_values