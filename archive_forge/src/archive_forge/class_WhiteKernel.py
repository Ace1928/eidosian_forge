import math
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from inspect import signature
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from ..base import clone
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import pairwise_kernels
from ..utils.validation import _num_samples
class WhiteKernel(StationaryKernelMixin, GenericKernelMixin, Kernel):
    """White kernel.

    The main use-case of this kernel is as part of a sum-kernel where it
    explains the noise of the signal as independently and identically
    normally-distributed. The parameter noise_level equals the variance of this
    noise.

    .. math::
        k(x_1, x_2) = noise\\_level \\text{ if } x_i == x_j \\text{ else } 0


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    noise_level : float, default=1.0
        Parameter controlling the noise level (variance)

    noise_level_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'noise_level'.
        If set to "fixed", 'noise_level' cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel(noise_level=0.5)
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1... ]), array([316.6..., 316.6...]))
    """

    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-05, 100000.0)):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter('noise_level', 'numeric', self.noise_level_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is not None and eval_gradient:
            raise ValueError('Gradient can only be evaluated when Y is None.')
        if Y is None:
            K = self.noise_level * np.eye(_num_samples(X))
            if eval_gradient:
                if not self.hyperparameter_noise_level.fixed:
                    return (K, self.noise_level * np.eye(_num_samples(X))[:, :, np.newaxis])
                else:
                    return (K, np.empty((_num_samples(X), _num_samples(X), 0)))
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.full(_num_samples(X), self.noise_level, dtype=np.array(self.noise_level).dtype)

    def __repr__(self):
        return '{0}(noise_level={1:.3g})'.format(self.__class__.__name__, self.noise_level)