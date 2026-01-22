from enum import Enum
from typing import Dict, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
from xformers.components import build_attention
from xformers.components.multi_head_dispatch import MultiHeadDispatchConfig
from xformers.factory import xFormer, xFormerConfig, xFormerEncoderConfig
from xformers.utils import generate_matching_config
class SCHead(nn.Module):

    def __init__(self, config, dim_embedding, dim_mlp):
        super().__init__()
        self.pooling = pooling(Pooling(config['pooling_mode']))
        self.mlpblock = nn.Sequential(nn.Linear(dim_embedding, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, config['common']['num_classes']))

    def forward(self, inp: torch.Tensor):
        seq_score = self.mlpblock(self.pooling(inp))
        return seq_score