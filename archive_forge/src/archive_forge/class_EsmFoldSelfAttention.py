import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
class EsmFoldSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, head_width, gated=False):
        super().__init__()
        assert embed_dim == num_heads * head_width
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width
        self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gated = gated
        if gated:
            self.g_proj = nn.Linear(embed_dim, embed_dim)
            torch.nn.init.zeros_(self.g_proj.weight)
            torch.nn.init.ones_(self.g_proj.bias)
        self.rescale_factor = self.head_width ** (-0.5)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, x, mask=None, bias=None, indices=None):
        """
        Basic self attention with optional mask and external pairwise bias. To handle sequences of different lengths,
        use mask.

        Inputs:
            x: batch of input sequneces (.. x L x C) mask: batch of boolean masks where 1=valid, 0=padding position (..
            x L_k) bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads)

        Outputs:
          sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """
        t = self.proj(x).view(*x.shape[:2], self.num_heads, -1)
        t = t.permute(0, 2, 1, 3)
        q, k, v = t.chunk(3, dim=-1)
        q = self.rescale_factor * q
        a = torch.einsum('...qc,...kc->...qk', q, k)
        if bias is not None:
            a = a + bias.permute(0, 3, 1, 2)
        if mask is not None:
            mask = mask[:, None, None]
            a = a.masked_fill(mask == False, -np.inf)
        a = nn.functional.softmax(a, dim=-1)
        y = torch.einsum('...hqk,...hkc->...qhc', a, v)
        y = y.reshape(*y.shape[:2], -1)
        if self.gated:
            y = self.g_proj(x).sigmoid() * y
        y = self.o_proj(y)
        return (y, a.permute(0, 3, 1, 2))