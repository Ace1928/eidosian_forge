import time
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy.special import expit  # logistic function
from ..base import (
from ..utils import check_random_state, gen_even_slices
from ..utils._param_validation import Interval
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted
def _sample_hiddens(self, v, rng):
    """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer.
        """
    p = self._mean_hiddens(v)
    return rng.uniform(size=p.shape) < p