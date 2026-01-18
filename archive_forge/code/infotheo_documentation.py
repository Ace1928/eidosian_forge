from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp

    Generalized cross-entropy measures.

    Parameters
    ----------
    px : array_like
        Discrete probability distribution of random variable X
    py : array_like
        Discrete probability distribution of random variable Y
    pxpy : 2d array_like, optional
        Joint probability distribution of X and Y.  If pxpy is None, X and Y
        are assumed to be independent.
    logbase : int or np.e, optional
        Default is 2 (bits)
    measure : str, optional
        The measure is the type of generalized cross-entropy desired. 'T' is
        the cross-entropy version of the Tsallis measure.  'CR' is Cressie-Read
        measure.
    