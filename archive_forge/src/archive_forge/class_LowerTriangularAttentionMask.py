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
class LowerTriangularAttentionMask(AttentionMask):
    """
    This is a lower triangular mask. This is common in decoder only models.
    This should reduce the computation and memory to roughly half as close to
    half of the mask is zero.

    The mask stays same for each head and each input.

    Nit pick (TODO) - While blocking, we need to ensure that the blocks along
    the diagonals are themselves lower triangular blocks. But, for performance
    measurement, this is ok to ignore as we treat the whole block as useful
    values.
    """

    def __init__(self, config=None):
        super(LowerTriangularAttentionMask, self).__init__(config)

    def gen_mask(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        return self.expand(causal_1d_pattern(seq_length))

    def __str__(self):
        return 'lower_triangular'