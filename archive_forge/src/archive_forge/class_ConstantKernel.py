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
class ConstantKernel(StationaryKernelMixin, GenericKernelMixin, Kernel):
    """Constant kernel.

    Can be used as part of a product-kernel where it scales the magnitude of
    the other factor (kernel) or as part of a sum-kernel, where it modifies
    the mean of the Gaussian process.

    .. math::
        k(x_1, x_2) = constant\\_value \\;\\forall\\; x_1, x_2

    Adding a constant kernel is equivalent to adding a constant::

            kernel = RBF() + ConstantKernel(constant_value=2)

    is the same as::

            kernel = RBF() + 2


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    constant_value : float, default=1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on `constant_value`.
        If set to "fixed", `constant_value` cannot be changed during
        hyperparameter tuning.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = RBF() + ConstantKernel(constant_value=2)
    >>> gpr = GaussianProcessRegressor(kernel=kernel, alpha=5,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3696...
    >>> gpr.predict(X[:1,:], return_std=True)
    (array([606.1...]), array([0.24...]))
    """

    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-05, 100000.0)):
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds

    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter('constant_value', 'numeric', self.constant_value_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object,             default=None
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

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),             optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError('Gradient can only be evaluated when Y is None.')
        K = np.full((_num_samples(X), _num_samples(Y)), self.constant_value, dtype=np.array(self.constant_value).dtype)
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (K, np.full((_num_samples(X), _num_samples(X), 1), self.constant_value, dtype=np.array(self.constant_value).dtype))
            else:
                return (K, np.empty((_num_samples(X), _num_samples(X), 0)))
        else:
            return K

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
        return np.full(_num_samples(X), self.constant_value, dtype=np.array(self.constant_value).dtype)

    def __repr__(self):
        return '{0:.3g}**2'.format(np.sqrt(self.constant_value))