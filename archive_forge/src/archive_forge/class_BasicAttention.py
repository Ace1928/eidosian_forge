import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
class BasicAttention(nn.Module):
    """
    Implements simple/classical attention.
    """

    def __init__(self, dim=1, attn='cosine', residual=False, get_weights=True):
        super().__init__()
        if attn == 'cosine':
            self.cosine = nn.CosineSimilarity(dim=dim)
        self.attn = attn
        self.dim = dim
        self.get_weights = get_weights
        self.residual = residual

    def forward(self, xs, ys, mask_ys=None, values=None):
        """
        Compute attention.

        Attend over ys with query xs to obtain weights, then apply weights to
        values (ys if yalues is None)

        Args:
            xs: B x query_len x dim (queries)
            ys: B x key_len x dim (keys)
            mask_ys: B x key_len (mask)
            values: B x value_len x dim (values); if None, default to ys
        """
        bsz = xs.size(0)
        y_len = ys.size(1)
        x_len = xs.size(1)
        if self.attn == 'cosine':
            l1 = self.cosine(xs, ys).unsqueeze(self.dim - 1)
        else:
            l1 = torch.bmm(xs, ys.transpose(1, 2))
            if self.attn == 'sqrt':
                d_k = ys.size(-1)
                l1 = l1 / math.sqrt(d_k)
        if mask_ys is not None:
            attn_mask = (mask_ys == 0).view(bsz, 1, y_len)
            attn_mask = attn_mask.repeat(1, x_len, 1)
            l1.masked_fill(attn_mask, neginf(l1.dtype))
        l2 = F.softmax(l1, dim=self.dim, dtype=torch.float).type_as(l1)
        if values is None:
            values = ys
        lhs_emb = torch.bmm(l2, values)
        if self.residual:
            lhs_emb = lhs_emb.add(xs)
        if self.get_weights:
            return (lhs_emb.squeeze(self.dim - 1), l2)
        else:
            return lhs_emb.squeeze(self.dim - 1)