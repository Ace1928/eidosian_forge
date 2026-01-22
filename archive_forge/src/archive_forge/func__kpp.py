from k-means models and quantizing vectors by comparing them with
import warnings
import numpy as np
from collections import deque
from scipy._lib._array_api import (
from scipy._lib._util import check_random_state, rng_integers
from scipy.spatial.distance import cdist
from . import _vq
def _kpp(data, k, rng, xp):
    """ Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    rng : `numpy.random.Generator` or `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    """
    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = xp.empty((int(k), dims))
    for i in range(k):
        if i == 0:
            init[i, :] = data[rng_integers(rng, data.shape[0]), :]
        else:
            D2 = cdist(init[:i, :], data, metric='sqeuclidean').min(axis=0)
            probs = D2 / D2.sum()
            cumprobs = probs.cumsum()
            r = rng.uniform()
            cumprobs = np.asarray(cumprobs)
            init[i, :] = data[np.searchsorted(cumprobs, r), :]
    return init