import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
class EncodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        embed = torch.zeros(config.codebook_size, config.codebook_dim)
        self.codebook_size = config.codebook_size
        self.register_buffer('inited', torch.Tensor([True]))
        self.register_buffer('cluster_size', torch.zeros(config.codebook_size))
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

    def quantize(self, hidden_states):
        embed = self.embed.t()
        scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
        dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def encode(self, hidden_states):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        embed_ind = self.quantize(hidden_states)
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    def decode(self, embed_ind):
        quantize = nn.functional.embedding(embed_ind, self.embed)
        return quantize