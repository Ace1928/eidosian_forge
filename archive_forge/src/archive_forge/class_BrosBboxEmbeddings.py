import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_bros import BrosConfig
class BrosBboxEmbeddings(nn.Module):

    def __init__(self, config):
        super(BrosBboxEmbeddings, self).__init__()
        self.bbox_sinusoid_emb = BrosPositionalEmbedding2D(config)
        self.bbox_projection = nn.Linear(config.dim_bbox_sinusoid_emb_2d, config.dim_bbox_projection, bias=False)

    def forward(self, bbox: torch.Tensor):
        bbox_t = bbox.transpose(0, 1)
        bbox_pos = bbox_t[None, :, :, :] - bbox_t[:, None, :, :]
        bbox_pos_emb = self.bbox_sinusoid_emb(bbox_pos)
        bbox_pos_emb = self.bbox_projection(bbox_pos_emb)
        return bbox_pos_emb