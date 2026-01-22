import logging
import sys
import time
import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.sparse import sparsetools
from gensim import interfaces, matutils, utils
from gensim.models import basemodel
from gensim.utils import is_empty
def ascarray(a, name=''):
    """Return a contiguous array in memory (C order).

    Parameters
    ----------
    a : numpy.ndarray
        Input array.
    name : str, optional
        Array name, used for logging purposes.

    Returns
    -------
    np.ndarray
        Contiguous array (row-major order) of same shape and content as `a`.

    """
    if not a.flags.contiguous:
        logger.debug('converting %s array %s to C order', a.shape, name)
        a = np.ascontiguousarray(a)
    return a