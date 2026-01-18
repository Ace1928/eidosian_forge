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
def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
    """Remove a subcluster from a node and update it with the
        split subclusters.
        """
    ind = self.subclusters_.index(subcluster)
    self.subclusters_[ind] = new_subcluster1
    self.init_centroids_[ind] = new_subcluster1.centroid_
    self.init_sq_norm_[ind] = new_subcluster1.sq_norm_
    self.append_subcluster(new_subcluster2)