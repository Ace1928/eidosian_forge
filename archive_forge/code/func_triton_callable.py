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
def triton_callable(self, a, b, mask, config):
    triton_kernel = self.get_triton_fn(mask, config, self.mode)
    input_a = sparsify_tensor(a, mask, config) if self.mode == 'spmm' else a
    input_b = b

    def triton_fn():
        return triton_kernel(input_a, input_b)
    return triton_fn