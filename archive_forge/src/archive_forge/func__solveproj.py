import collections.abc
import logging
import numpy as np
import scipy.sparse
from scipy.stats import halfnorm
from gensim import interfaces
from gensim import matutils
from gensim import utils
from gensim.interfaces import TransformedCorpus
from gensim.models import basemodel, CoherenceModel
from gensim.models.nmf_pgd import solve_h
def _solveproj(self, v, W, h=None, v_max=None):
    """Update residuals and representation (h) matrices.

        Parameters
        ----------
        v : scipy.sparse.csc_matrix
            Subset of training corpus.
        W : ndarray
            Dictionary matrix.
        h : ndarray
            Representation matrix.
        v_max : float
            Maximum possible value in matrices.

        """
    m, n = W.shape
    if v_max is not None:
        self.v_max = v_max
    elif self.v_max is None:
        self.v_max = v.max()
    batch_size = v.shape[1]
    hshape = (n, batch_size)
    if h is None or h.shape != hshape:
        h = np.zeros(hshape)
    Wt = W.T
    WtW = Wt.dot(W)
    h_error = None
    for iter_number in range(self._h_max_iter):
        logger.debug('h_error: %s', h_error)
        Wtv = self._dense_dot_csc(Wt, v)
        permutation = self.random_state.permutation(self.num_topics).astype(np.int32)
        error_ = solve_h(h, Wtv, WtW, permutation, self._kappa)
        error_ /= m
        if h_error and np.abs(h_error - error_) < self._h_stop_condition:
            break
        h_error = error_
    return h