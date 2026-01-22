import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
def _compute_distances(self, x1, x2):
    """
        This function computes the euclidean distance between every vector
        in the two batches in input.
        """
    assert x1.shape == x2.shape
    batch_size, dim = x1.shape
    x1_ = x1.expand_dims(1).broadcast_to([batch_size, batch_size, dim])
    x2_ = x2.expand_dims(0).broadcast_to([batch_size, batch_size, dim])
    squared_diffs = (x1_ - x2_) ** 2
    return squared_diffs.sum(axis=2)