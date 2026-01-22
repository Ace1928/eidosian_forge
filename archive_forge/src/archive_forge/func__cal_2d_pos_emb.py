import collections
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
def _cal_2d_pos_emb(self, bbox):
    position_coord_x = bbox[:, :, 0]
    position_coord_y = bbox[:, :, 3]
    rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
    rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
    rel_pos_x = self.relative_position_bucket(rel_pos_x_2d_mat, num_buckets=self.rel_2d_pos_bins, max_distance=self.max_rel_2d_pos)
    rel_pos_y = self.relative_position_bucket(rel_pos_y_2d_mat, num_buckets=self.rel_2d_pos_bins, max_distance=self.max_rel_2d_pos)
    rel_pos_x = self.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
    rel_pos_y = self.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
    rel_pos_x = rel_pos_x.contiguous()
    rel_pos_y = rel_pos_y.contiguous()
    rel_2d_pos = rel_pos_x + rel_pos_y
    return rel_2d_pos