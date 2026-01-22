import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig
def _prepare_attention_mask(self, input_ids, span, context, length):
    batch = input_ids.size(0)
    seqlen = input_ids.size(1)
    device = input_ids.device
    directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)
    attention_mask = context[:, None, :] | context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
    attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
    mask_1d = torch.tensor(list(range(seqlen - self.prompt_length))[::-1], device=device)[None, :].repeat(batch, 1) < length[:, None]
    mask_1d = torch.cat((torch.ones(batch, self.prompt_length, device=device).bool(), mask_1d), dim=1)
    attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
    return attention_mask