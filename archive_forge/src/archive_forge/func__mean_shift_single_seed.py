import warnings
from collections import defaultdict
from numbers import Integral, Real
import numpy as np
from .._config import config_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..metrics.pairwise import pairwise_distances_argmin
from ..neighbors import NearestNeighbors
from ..utils import check_array, check_random_state, gen_batches
from ..utils._param_validation import Interval, validate_params
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 0.001 * bandwidth
    completed_iterations = 0
    while True:
        i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth, return_distance=False)[0]
        points_within = X[i_nbrs]
        if len(points_within) == 0:
            break
        my_old_mean = my_mean
        my_mean = np.mean(points_within, axis=0)
        if np.linalg.norm(my_mean - my_old_mean) < stop_thresh or completed_iterations == max_iter:
            break
        completed_iterations += 1
    return (tuple(my_mean), len(points_within), completed_iterations)