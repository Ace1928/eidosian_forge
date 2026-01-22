import copy
import itertools
import operator
import string
import warnings
import cupy
from cupy._core import _accelerator
from cupy import _util
from cupy.linalg._einsum_opt import _greedy_path
from cupy.linalg._einsum_opt import _optimal_path
from cupy.linalg._einsum_cutn import _try_use_cutensornet
def _iter_path_pairs(path):
    """Decompose path into binary path

    Args:
        path (sequence of tuples of ints)

    Yields:
        tuple of ints: pair (idx0, idx1) that represents the operation
            {pop(idx0); pop(idx1); append();}
    """
    for indices in path:
        assert all((idx >= 0 for idx in indices))
        if len(indices) >= 2:
            indices = sorted(indices, reverse=True)
            yield (indices[0], indices[1])
            for idx in indices[2:]:
                yield (-1, idx)