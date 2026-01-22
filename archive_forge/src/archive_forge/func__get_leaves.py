import warnings
from math import sqrt
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import pairwise_distances_argmin
from ..metrics.pairwise import euclidean_distances
from ..utils._param_validation import Interval
from ..utils.extmath import row_norms
from ..utils.validation import check_is_fitted
from . import AgglomerativeClustering
def _get_leaves(self):
    """
        Retrieve the leaves of the CF Node.

        Returns
        -------
        leaves : list of shape (n_leaves,)
            List of the leaf nodes.
        """
    leaf_ptr = self.dummy_leaf_.next_leaf_
    leaves = []
    while leaf_ptr is not None:
        leaves.append(leaf_ptr)
        leaf_ptr = leaf_ptr.next_leaf_
    return leaves