import dataclasses
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_pegasus_x import PegasusXConfig
@classmethod
def unpad_local_tokens(cls, padded_hidden_states, block_size):
    pad_size = block_size // 2
    return padded_hidden_states[:, pad_size:-pad_size, :]