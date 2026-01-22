from enum import Enum
from typing import Dict, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
from xformers.components import build_attention
from xformers.components.multi_head_dispatch import MultiHeadDispatchConfig
from xformers.factory import xFormer, xFormerConfig, xFormerEncoderConfig
from xformers.utils import generate_matching_config
def append_cls(inp, mask, vocab_size):
    batch_size = inp.size(0)
    cls_id = ((vocab_size - 1) * torch.ones(batch_size, dtype=torch.long, device=inp.device)).long()
    cls_mask = torch.ones(batch_size, dtype=torch.float, device=mask.device)
    inp = torch.cat([cls_id[:, None], inp[:, :-1]], dim=-1)
    mask = torch.cat([cls_mask[:, None], mask[:, :-1]], dim=-1)
    return (inp, mask)