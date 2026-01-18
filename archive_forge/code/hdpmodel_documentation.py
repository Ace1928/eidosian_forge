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
Format the display for a single topic in two different ways.

        Parameters
        ----------
        topic_id : int
            Acts as a representative index for a particular topic.
        topic_terms : list of (str, numpy.float)
            Contains the most probable words from a single topic.

        Returns
        -------
        list of (str, numpy.float) **or** list of str
            Output format for topic terms depends on the value of `self.style` attribute.

        