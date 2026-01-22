import warnings
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..metrics.pairwise import _VALID_METRICS
from ..neighbors import NearestNeighbors
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import _check_sample_weight
from ._dbscan_inner import dbscan_inner
Compute clusters from a data or distance matrix and predict labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or             (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        