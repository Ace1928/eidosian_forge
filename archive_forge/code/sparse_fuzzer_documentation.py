from typing import Optional, Tuple, Union
from numbers import Number
import torch
from torch.utils.benchmark import FuzzedTensor
import math
sparse_tensor_constructor creates a sparse tensor with coo format.

        Note that when `is_coalesced` is False, the number of elements is doubled but the number of indices
        represents the same amount of number of non zeros `nnz`, i.e, this is virtually the same tensor
        with the same sparsity pattern. Moreover, most of the sparse operation will use coalesce() method
        and what we want here is to get a sparse tensor with the same `nnz` even if this is coalesced or not.

        In the other hand when `is_coalesced` is True the number of elements is reduced in the coalescing process
        by an unclear amount however the probability to generate duplicates indices are low for most of the cases.
        This decision was taken on purpose to maintain the construction cost as low as possible.
        