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
class KernelOperator(Kernel):
    """Base class for all kernel operators.

    .. versionadded:: 0.18
    """

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict(k1=self.k1, k2=self.k2)
        if deep:
            deep_items = self.k1.get_params().items()
            params.update((('k1__' + k, val) for k, val in deep_items))
            deep_items = self.k2.get_params().items()
            params.update((('k2__' + k, val) for k, val in deep_items))
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = [Hyperparameter('k1__' + hyperparameter.name, hyperparameter.value_type, hyperparameter.bounds, hyperparameter.n_elements) for hyperparameter in self.k1.hyperparameters]
        for hyperparameter in self.k2.hyperparameters:
            r.append(Hyperparameter('k2__' + hyperparameter.name, hyperparameter.value_type, hyperparameter.bounds, hyperparameter.n_elements))
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return np.append(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        k1_dims = self.k1.n_dims
        self.k1.theta = theta[:k1_dims]
        self.k2.theta = theta[k1_dims:]

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        if self.k1.bounds.size == 0:
            return self.k2.bounds
        if self.k2.bounds.size == 0:
            return self.k1.bounds
        return np.vstack((self.k1.bounds, self.k2.bounds))

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return self.k1 == b.k1 and self.k2 == b.k2 or (self.k1 == b.k2 and self.k2 == b.k1)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.k1.is_stationary() and self.k2.is_stationary()

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is stationary."""
        return self.k1.requires_vector_input or self.k2.requires_vector_input