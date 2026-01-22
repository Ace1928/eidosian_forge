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

    In this experiment, we analyze how increasing the block size affects
    performance.  We will take the lower triangular pattern. As we increase the
    batch size, the blocks near the diagonal have to do more unnecessary
    computation (the effective sparsity starts decreasing).
    