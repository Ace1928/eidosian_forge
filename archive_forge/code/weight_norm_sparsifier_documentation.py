from functools import reduce
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from .base_sparsifier import BaseSparsifier
Creates a block-level mask.

        Block-level mask is described as a mask, where the granularity of sparsification of the
        largest patch is the sparse_block_shape. That means that for a given mask and a
        sparse_block_shape, the sparsity is computed only within a patch of a size sparse_block_shape.

        In this context the `zeros_per_block` describes the number of zeroed-out elements within a patch.
        