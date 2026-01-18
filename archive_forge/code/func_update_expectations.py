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
def update_expectations(self):
    """Since we're doing lazy updates on lambda, at any given moment the current state of lambda may not be
        accurate. This function updates all of the elements of lambda and Elogbeta so that if (for example) we want to
        print out the topics we've learned we'll get the correct behavior.

        """
    for w in range(self.m_W):
        self.m_lambda[:, w] *= np.exp(self.m_r[-1] - self.m_r[self.m_timestamp[w]])
    self.m_Elogbeta = psi(self.m_eta + self.m_lambda) - psi(self.m_W * self.m_eta + self.m_lambda_sum[:, np.newaxis])
    self.m_timestamp[:] = self.m_updatect
    self.m_status_up_to_date = True