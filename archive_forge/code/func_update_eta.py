import logging
import numbers
import os
import time
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from gensim import interfaces, utils, matutils
from gensim.matutils import (
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback
def update_eta(self, lambdat, rho):
    """Update parameters for the Dirichlet prior on the per-topic word weights.

        Parameters
        ----------
        lambdat : numpy.ndarray
            Previous lambda parameters.
        rho : float
            Learning rate.

        Returns
        -------
        numpy.ndarray
            The updated eta parameters.

        """
    N = float(lambdat.shape[0])
    logphat = (sum((dirichlet_expectation(lambda_) for lambda_ in lambdat)) / N).reshape((self.num_terms,))
    assert logphat.dtype == self.dtype
    self.eta = update_dir_prior(self.eta, N, logphat, rho)
    assert self.eta.dtype == self.dtype
    return self.eta