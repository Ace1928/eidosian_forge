import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerTransformerDecoderQueryTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=False, layer_norm_eps=1e-05):
        super().__init__()
        decoder_layer = OneFormerTransformerDecoderQueryTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, layer_norm_eps)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = OneFormerTransformerDecoderQueryTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, mask, query_embed, pos_embed, task_token=None):
        batch_size = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        if mask is not None:
            mask = mask.flatten(1)
        if task_token is None:
            queries = torch.zeros_like(query_embed)
        else:
            queries = task_token.repeat(query_embed.shape[0], 1, 1)
        queries = self.decoder(queries, src, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return queries.transpose(1, 2)