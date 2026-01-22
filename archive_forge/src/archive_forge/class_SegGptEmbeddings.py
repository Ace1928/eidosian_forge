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
class SegGptEmbeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings for input and prompt.
    """

    def __init__(self, config: SegGptConfig) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.segment_token_input = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.segment_token_prompt = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.type_token_semantic = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.type_token_instance = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        self.patch_embeddings = SegGptPatchEmbeddings(config)
        num_positions = (config.pretrain_image_size // config.patch_size) ** 2 + 1
        self.position_embeddings = nn.Parameter(torch.randn(1, num_positions, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        patch_pos_embed = self.position_embeddings[:, 1:]
        num_patches = patch_pos_embed.shape[1]
        pretrain_patch_size = int(math.sqrt(num_patches))
        if pretrain_patch_size != height or pretrain_patch_size != width:
            patch_pos_embed = F.interpolate(patch_pos_embed.reshape(1, pretrain_patch_size, pretrain_patch_size, -1).permute(0, 3, 1, 2), size=(height, width), mode='bicubic', align_corners=False)
            return patch_pos_embed.permute(0, 2, 3, 1)
        else:
            return patch_pos_embed.reshape(1, height, width, -1)

    def forward(self, pixel_values: torch.Tensor, prompt_pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor]=None, embedding_type: Optional[str]=None) -> torch.Tensor:
        input_embeddings = self.patch_embeddings(pixel_values)
        prompt_embeddings = self.patch_embeddings(prompt_pixel_values)
        batch_size, patch_height, patch_width, _ = input_embeddings.shape
        mask_token = self.mask_token.expand(batch_size, patch_height, patch_width, -1)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, patch_height, patch_width, 1)
        prompt_embeddings = prompt_embeddings * (1 - w) + mask_token * w
        embedding_type = embedding_type if embedding_type is not None else 'instance'
        pos_embed = self.interpolate_pos_encoding(patch_height, patch_width)
        input_embeddings = input_embeddings + self.segment_token_input
        prompt_embeddings = prompt_embeddings + self.segment_token_prompt
        input_embeddings = input_embeddings + pos_embed
        prompt_embeddings = prompt_embeddings + pos_embed
        if embedding_type == 'semantic':
            type_embedding = self.type_token_semantic
        elif embedding_type == 'instance':
            type_embedding = self.type_token_instance
        else:
            raise ValueError(f"Embedding type should be either 'semantic' or 'instance', but got {embedding_type}")
        input_embeddings = input_embeddings + type_embedding
        prompt_embeddings = prompt_embeddings + type_embedding
        embeddings = torch.cat((input_embeddings, prompt_embeddings), dim=0)
        return embeddings