import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
class TvltAudioEmbeddings(nn.Module):
    """Construct the patch and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = TvltAudioPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        self.type_embed_a = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.num_freq_patches = config.frequency_length // config.audio_patch_size[1]
        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.num_patches // self.num_freq_patches, config.hidden_size))
        self.freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_patches, config.hidden_size))
        self.num_freq_patches = config.frequency_length // config.audio_patch_size[1]
        self.config = config

    def forward(self, audio_values, attention_masks=None):
        embeddings = self.patch_embeddings(audio_values)
        num_time_patches = embeddings.size(1) // self.num_freq_patches
        embeddings += self.freq_embed.repeat(1, num_time_patches, 1)
        embeddings += torch.repeat_interleave(self.pos_embed_a[:, :num_time_patches], self.num_freq_patches, dim=1)
        embeddings += self.type_embed_a
        return (embeddings, attention_masks)