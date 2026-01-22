from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import linalg
class GMM(object):
    """
    Gaussian Mixture Model

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Initializes parameters such that every mixture component has zero
    mean and identity covariance.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.
    covariance_type : {'diag', 'spherical', 'tied', 'full'}
        String describing the type of covariance parameters to
        use. Defaults to 'diag'.

    Attributes
    ----------
    `weights_` : array, shape (n_components,)
        This attribute stores the mixing weights for each mixture component.
    `means_` : array, shape (n_components, n_features)
        Mean parameters for each mixture component.
    `covars_` : array
        Covariance parameters for each mixture component. The shape
        depends on `covariance_type`.::

        - (n_components, n_features)             if 'spherical',
        - (n_features, n_features)               if 'tied',
        - (n_components, n_features)             if 'diag',
        - (n_components, n_features, n_features) if 'full'.

    `converged_` : bool
        True when convergence was reached in fit(), False otherwise.

    See Also
    --------
    sklearn.mixture.GMM

    """

    def __init__(self, n_components=1, covariance_type='full'):
        if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('Invalid value for covariance_type: %s' % covariance_type)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = None
        self.covars = None

    def __setstate__(self, state):
        try:
            import warnings
            warnings.warn('Please update your GMM models by loading them and saving them again. Loading old models will not work from version 0.16 onwards.')
            state['weights'] = state.pop('weights_')
            state['means'] = state.pop('means_')
            state['covars'] = state.pop('covars_')
        except KeyError:
            pass
        self.__dict__.update(state)

    def score_samples(self, x):
        """
        Return the per-sample likelihood of the data under the model.

        Compute the log probability of x under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of x.

        Parameters
        ----------
        x: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds
            to a single data point.

        Returns
        -------
        log_prob : array_like, shape (n_samples,)
            Log probabilities of each data point in `x`.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation.

        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if x.size == 0:
            return (np.array([]), np.empty((0, self.n_components)))
        if x.shape[1] != self.means.shape[1]:
            raise ValueError('The shape of x is not compatible with self')
        lpr = log_multivariate_normal_density(x, self.means, self.covars, self.covariance_type) + np.log(self.weights)
        log_prob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - log_prob[:, np.newaxis])
        return (log_prob, responsibilities)

    def score(self, x):
        """
        Compute the log probability under the model.

        Parameters
        ----------
        x : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array_like, shape (n_samples,)
            Log probabilities of each data point in `x`.

        """
        log_prob, _ = self.score_samples(x)
        return log_prob

    def fit(self, x, random_state=None, tol=0.001, min_covar=0.001, n_iter=100, n_init=1, params='wmc', init_params='wmc'):
        """
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

        """
        import sklearn.mixture
        gmm = sklearn.mixture.GMM(n_components=self.n_components, covariance_type=self.covariance_type, random_state=random_state, tol=tol, min_covar=min_covar, n_iter=n_iter, n_init=n_init, params=params, init_params=init_params)
        gmm.fit(x)
        self.covars = gmm.covars_
        self.means = gmm.means_
        self.weights = gmm.weights_
        return self