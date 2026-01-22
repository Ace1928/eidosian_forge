import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
def _bigbird_block_rand_mask(self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1):
    """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
    if from_seq_length // from_block_size != to_seq_length // to_block_size:
        raise ValueError('Error the number of blocks needs to be same!')
    rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
    if not self.training:
        return rand_attn
    middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = to_seq_length // to_block_size - 1
    if last_idx > 2 * to_block_size:
        last = last_idx // to_block_size - 1
    r = num_rand_blocks
    for i in range(1, from_seq_length // from_block_size - 1):
        start = i - 2
        end = i
        if i == 1:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_seq_length // from_block_size - 3:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
        elif i == from_seq_length // from_block_size - 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
        elif start > last:
            start = last
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
        elif end + 1 == last:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
        else:
            rand_attn[i - 1, :] = np.random.permutation(np.concatenate((middle_seq[:start], middle_seq[end + 1:last])))[:r]
    return rand_attn