from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import linalg

        Estimate model parameters with the expectation-maximization algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating the
        GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        x : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row corresponds
            to a single data point.
        random_state: RandomState or an int seed (0 by default)
            A random number generator instance.
        min_covar : float, optional
            Floor on the diagonal of the covariance matrix to prevent
            overfitting.
        tol : float, optional
            Convergence threshold. EM iterations will stop when average
            gain in log-likelihood is below this threshold.
        n_iter : int, optional
            Number of EM iterations to perform.
        n_init : int, optional
            Number of initializations to perform, the best results is kept.
        params : str, optional
            Controls which parameters are updated in the training process.
            Can contain any combination of 'w' for weights, 'm' for means,
            and 'c' for covars.
        init_params : str, optional
            Controls which parameters are updated in the initialization
            process.  Can contain any combination of 'w' for weights,
            'm' for means, and 'c' for covars.

        