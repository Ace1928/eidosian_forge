import warnings
import numpy as np
import scipy.sparse as sp
from ..base import _fit_context
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Integral, Interval, StrOptions
from ..utils.extmath import row_norms
from ..utils.validation import _check_sample_weight, check_is_fitted, check_random_state
from ._k_means_common import _inertia_dense, _inertia_sparse
from ._kmeans import (
Predict recursively by going down the hierarchical tree.

        Parameters
        ----------
        X : {ndarray, csr_matrix} of shape (n_samples, n_features)
            The data points, currently assigned to `cluster_node`, to predict between
            the subclusters of this node.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X.

        cluster_node : _BisectingTree node object
            The cluster node of the hierarchical tree.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        