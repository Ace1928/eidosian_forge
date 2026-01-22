import gc
import math
from collections import namedtuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import pretty_barplot
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import SparseCS, _matmul_with_mask
class BigBirdAttentionMask(AttentionMask):
    """
    BigBird mask are composed of three types of masks - random, global and window.
    For more details, refer to https://arxiv.org/pdf/2007.14062.pdf

    One point to note is that mask is per head here. So, mask is 3D tensor.
    (num_heads, seq_length, seq_length).
    """

    def __init__(self, config=None):
        super(BigBirdAttentionMask, self).__init__(config)
        self.mask_per_head = True
        self.set_config_attr('num_global_tokens', 2 * self.config.block_size)
        self.set_config_attr('num_random_tokens', 3 * self.config.block_size)
        self.set_config_attr('num_window_tokens', 3 * self.config.block_size)

    def gen_global_mask(self, seq_length):
        num_global_blocks = self.config.num_global_tokens // self.config.block_size
        mask_indices = torch.randint(0, seq_length - 1, size=(num_global_blocks,))
        mask_indices = torch.unique(mask_indices)
        query_mask = torch.zeros(seq_length).to(dtype=torch.bool)
        query_mask.scatter_(0, mask_indices, True)
        return global_token_pattern(query_mask)

    def gen_random_mask(self, seq_length):
        num_random_blocks = self.config.num_random_tokens // self.config.block_size
        mask_indices = torch.randint(0, seq_length - 1, size=(seq_length, num_random_blocks))
        random_mask = torch.zeros(seq_length, seq_length).to(dtype=torch.bool)
        random_mask.scatter_(1, mask_indices, True)
        return random_mask

    def gen_window_mask(self, seq_length):
        num_window_blocks = self.config.num_window_tokens // self.config.block_size
        if num_window_blocks % 2 == 0:
            num_window_blocks += 1
        return local_1d_pattern(seq_length, num_window_blocks)

    def gen_mask(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        assert keep_blocked, 'Not implemented, call to_dense later to get full tensor'
        if self.mask_per_head:
            head_masks = []
            for _ in range(self.config.num_heads):
                global_mask = self.gen_global_mask(seq_length)
                random_mask = self.gen_random_mask(seq_length)
                window_mask = self.gen_window_mask(seq_length)
                mask = global_mask + random_mask + window_mask
                head_masks.append(mask)
            mask = torch.stack(head_masks)
        else:
            global_mask = self.gen_global_mask(seq_length)
            random_mask = self.gen_random_mask(seq_length)
            window_mask = self.gen_window_mask(seq_length)
            mask = global_mask + random_mask + window_mask
            mask = self.expand(mask)
        return mask

    def __str__(self):
        return 'bigbird'