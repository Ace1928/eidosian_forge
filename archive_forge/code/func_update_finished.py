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
def update_finished(self, start_time, chunks_processed, docs_processed):
    """Flag to determine whether the model has been updated with the new corpus or not.

        Parameters
        ----------
        start_time : float
            Indicates the current processor time as a floating point number expressed in seconds.
            The resolution is typically better on Windows than on Unix by one microsecond due to differing
            implementation of underlying function calls.
        chunks_processed : int
            Indicates progress of the update in terms of the number of chunks processed.
        docs_processed : int
            Indicates number of documents finished processing.This is incremented in size of chunks.

        Returns
        -------
        bool
            If True - model is updated, False otherwise.

        """
    return self.max_chunks and chunks_processed == self.max_chunks or (self.max_time and time.perf_counter() - start_time > self.max_time) or (not self.max_chunks and (not self.max_time) and (docs_processed >= self.m_D))