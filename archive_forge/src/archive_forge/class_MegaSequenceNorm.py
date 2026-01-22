import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaSequenceNorm(nn.Module):
    """
    A wrapper class for various layer normalization options used in Mega. Used to handle differences in expectations on
    input axis locations for different normalization methods.
    """

    def __init__(self, norm_type, embedding_dim, eps=1e-05, affine=True, export=False):
        super().__init__()
        if norm_type == 'layernorm':
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=affine)
        elif norm_type == 'scalenorm':
            self.norm = MegaScaleNorm(dim=-1, eps=eps, affine=affine)
        elif norm_type == 'rmsnorm':
            self.norm = MegaRMSNorm(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(embedding_dim, eps=eps, affine=affine)
        elif norm_type == 'syncbatchnorm':
            self.norm = nn.SyncBatchNorm(embedding_dim, eps=eps, affine=affine)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def forward(self, input):
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            if input.dim() != 3:
                raise ValueError('BatchNorm inputs must be exactly 3-dimensional')
            input = input.permute(1, 2, 0)
            input = self.norm(input)
            return input.permute(2, 0, 1)
        else:
            return self.norm(input)