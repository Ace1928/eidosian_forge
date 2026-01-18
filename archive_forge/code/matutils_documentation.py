from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
Expected value of log(theta) where theta is drawn from a Dirichlet distribution.

        Parameters
        ----------
        alpha : numpy.ndarray
            Dirichlet parameter 2d matrix or 1d vector, if 2d - each row is treated as a separate parameter vector.

        Returns
        -------
        numpy.ndarray
            Log of expected values, dimension same as `alpha.ndim`.

        