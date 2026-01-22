import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
    proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
    proj_relative_position_embeddings = proj_relative_position_embeddings.view(relative_position_embeddings.size(0), -1, self.num_heads, self.head_size)
    proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
    proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)
    query = query.transpose(1, 2)
    q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
    q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)
    scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))
    scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)
    zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
    scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
    scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
    scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
    scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
    scores_bd = scores_bd[:, :, :, :scores_bd.size(-1) // 2 + 1]
    scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)
    return scores