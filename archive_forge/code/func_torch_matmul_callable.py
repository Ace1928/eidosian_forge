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
def torch_matmul_callable(self, a, b, mask, config):
    input_a = mask_tensor(a, mask, config) if self.mode == 'spmm' else a
    input_b = b.transpose(-1, -2) if self.mode == 'sddmm' else b

    def torch_fn():
        return torch.matmul(input_a, input_b)
    return torch_fn