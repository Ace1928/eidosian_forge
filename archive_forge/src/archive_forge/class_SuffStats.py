from `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical Dirichlet Process",  JMLR (2011)
import logging
import time
import warnings
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.models import basemodel, ldamodel
from gensim.utils import deprecated
class SuffStats:
    """Stores sufficient statistics for the current chunk of document(s) whenever Hdp model is updated with new corpus.
    These stats are used when updating lambda and top level sticks. The statistics include number of documents in the
    chunk, length of words in the documents and top level truncation level.

    """

    def __init__(self, T, Wt, Dt):
        """

        Parameters
        ----------
        T : int
            Top level truncation level.
        Wt : int
            Length of words in the documents.
        Dt : int
            Chunk size.

        """
        self.m_chunksize = Dt
        self.m_var_sticks_ss = np.zeros(T)
        self.m_var_beta_ss = np.zeros((T, Wt))

    def set_zero(self):
        """Fill the sticks and beta array with 0 scalar value."""
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)