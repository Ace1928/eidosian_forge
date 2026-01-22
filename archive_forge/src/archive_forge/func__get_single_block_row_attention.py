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
@staticmethod
def _get_single_block_row_attention(block_id, to_start_block_id, to_end_block_id, num_rand_blocks, window_block_left=1, window_block_right=1, global_block_left=1, global_block_right=1):
    """
        For a single row block get random row attention.

        Args:
            block_id: int. block id of row.
            to_start_block_id: int. random attention column start id.
            to_end_block_id: int. random attention column end id.
            num_rand_blocks: int. number of random blocks to be selected.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            row containing the random attention vector of size num_rand_blocks.
        """
    to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
    perm_block = np.random.permutation(to_block_list)
    illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))
    illegal_blocks.extend(list(range(global_block_left)))
    illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))
    if block_id == 1:
        illegal_blocks.append(to_end_block_id - 2)
    if block_id == to_end_block_id - 2:
        illegal_blocks.append(1)
    selected_random_blokcs = []
    for i in range(to_end_block_id - to_start_block_id):
        if perm_block[i] not in illegal_blocks:
            selected_random_blokcs.append(perm_block[i])
        if len(selected_random_blokcs) == num_rand_blocks:
            break
    return np.array(selected_random_blokcs, dtype=np.int32)