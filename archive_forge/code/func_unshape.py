import copy
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.generation import GenerationConfig
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_pop2piano import Pop2PianoConfig
def unshape(states):
    """reshape"""
    return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)