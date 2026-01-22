import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax
class RelPartialLearnableDecoderLayer(nn.Module):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-05, **kwargs):
        super().__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'), layer_norm_epsilon=layer_norm_epsilon)

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None, output_attentions=False):
        attn_outputs = self.dec_attn(dec_inp, r, attn_mask=dec_attn_mask, mems=mems, head_mask=head_mask, output_attentions=output_attentions)
        ff_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output] + attn_outputs[1:]
        return outputs