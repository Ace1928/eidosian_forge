import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _rebuild_sparse_tensor(layout, data):
    """
    Rebuilds a sparse tensor from its sparse storage representation.

    Args:
        layout (str): The sparse storage layout of the tensor.
        data (tuple): The tensor's sparse storage representation.
    """
    if layout == torch.sparse_coo:
        if len(data) == 3:
            indices, values, size = data
            is_coalesced = None
        else:
            indices, values, size, is_coalesced = data
        result = torch.sparse_coo_tensor(indices, values, size, check_invariants=False, is_coalesced=is_coalesced)
        _sparse_tensors_to_validate.append(result)
        return result
    elif layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
        compressed_indices, plain_indices, values, size = data
        result = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, size, layout=layout, check_invariants=False)
        _sparse_tensors_to_validate.append(result)
        return result
    raise NotImplementedError(f'rebuilding sparse tensor for layout {layout}')