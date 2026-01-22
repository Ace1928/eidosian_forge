import copy
import math
import warnings
from typing import Any, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longt5 import LongT5Config
def _make_global_fixed_block_ids(attention_mask: torch.Tensor, global_block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the "fixed block" global id corresponding to each input token.

    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    In our scenario, as we use this strategy only for a decoder, orphan tokens, i.e. those tokens which do not make for
    the whole fixed block, are assigned to the preceding block.

    Padding tokens from the original sequence are represented by -1.
    """
    batch_size, seq_len = attention_mask.shape[:2]

    def handle_orphan_tokens(block_ids: torch.Tensor) -> torch.Tensor:
        block_ends = torch.arange(seq_len) % global_block_size == global_block_size - 1
        block_ends = block_ends.to(block_ids.device)
        true_block_ends = torch.logical_and(block_ends, block_ids >= 0)
        full_blocks = true_block_ends.sum(-1).unsqueeze(-1).type(block_ids.dtype) - 1
        block_ids = torch.where(block_ids < full_blocks, block_ids, full_blocks)
        return block_ids
    fixed_block_mask = torch.ones_like(attention_mask, device=attention_mask.device) / global_block_size
    fixed_block_mask = torch.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    mask = torch.where(attention_mask != 0.0, 1.0, -1000.0).type(attention_mask.dtype)
    global_block_ids = torch.floor(mask + fixed_block_mask - 1.0).type(attention_mask.dtype)
    _global_block_ids_lower_bound = torch.tensor(-1, dtype=global_block_ids.dtype, device=global_block_ids.device)
    global_block_ids = torch.where(global_block_ids > _global_block_ids_lower_bound, global_block_ids, _global_block_ids_lower_bound)
    global_block_ids = global_block_ids * attention_mask + (attention_mask - 1)
    global_block_ids = handle_orphan_tokens(global_block_ids)
    num_globals = seq_len // global_block_size
    if num_globals > 0:
        _sequence_block_ids_max = torch.max(global_block_ids, dim=-1).values.repeat(num_globals, 1).transpose(0, 1)
    else:
        _sequence_block_ids_max = torch.zeros(batch_size, 0, dtype=global_block_ids.dtype, device=global_block_ids.device)
    global_segment_ids = torch.cumsum(torch.ones(batch_size, num_globals), dim=-1) - 1
    global_segment_ids = global_segment_ids.to(attention_mask.device)
    global_segment_ids = torch.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)
    return (global_block_ids.type(torch.int), global_segment_ids.type(torch.int))