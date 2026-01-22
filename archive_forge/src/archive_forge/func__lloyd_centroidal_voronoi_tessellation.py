from __future__ import annotations
import copy
import math
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
import numpy as np
import scipy.stats as stats
from scipy._lib._util import rng_integers, _rng_spawn
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance, Voronoi
from scipy.special import gammainc
from ._sobol import (
from ._qmc_cy import (
def _lloyd_centroidal_voronoi_tessellation(sample: npt.ArrayLike, *, tol: DecimalNumber=1e-05, maxiter: IntNumber=10, qhull_options: str | None=None, **kwargs: dict) -> np.ndarray:
    """Approximate Centroidal Voronoi Tessellation.

    Perturb samples in N-dimensions using Lloyd-Max algorithm.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to iterate on. With ``n`` the number of samples and ``d``
        the dimension. Samples must be in :math:`[0, 1]^d`, with ``d>=2``.
    tol : float, optional
        Tolerance for termination. If the min of the L1-norm over the samples
        changes less than `tol`, it stops the algorithm. Default is 1e-5.
    maxiter : int, optional
        Maximum number of iterations. It will stop the algorithm even if
        `tol` is above the threshold.
        Too many iterations tend to cluster the samples as a hypersphere.
        Default is 10.
    qhull_options : str, optional
        Additional options to pass to Qhull. See Qhull manual
        for details. (Default: "Qbb Qc Qz Qj Qx" for ndim > 4 and
        "Qbb Qc Qz Qj" otherwise.)

    Returns
    -------
    sample : array_like (n, d)
        The sample after being processed by Lloyd-Max algorithm.

    Notes
    -----
    Lloyd-Max algorithm is an iterative process with the purpose of improving
    the dispersion of samples. For given sample: (i) compute a Voronoi
    Tessellation; (ii) find the centroid of each Voronoi cell; (iii) move the
    samples toward the centroid of their respective cell. See [1]_, [2]_.

    A relaxation factor is used to control how fast samples can move at each
    iteration. This factor is starting at 2 and ending at 1 after `maxiter`
    following an exponential decay.

    The process converges to equally spaced samples. It implies that measures
    like the discrepancy could suffer from too many iterations. On the other
    hand, L1 and L2 distances should improve. This is especially true with
    QMC methods which tend to favor the discrepancy over other criteria.

    .. note::

        The current implementation does not intersect the Voronoi Tessellation
        with the boundaries. This implies that for a low number of samples,
        empirically below 20, no Voronoi cell is touching the boundaries.
        Hence, samples cannot be moved close to the boundaries.

        Further improvements could consider the samples at infinity so that
        all boundaries are segments of some Voronoi cells. This would fix
        the computation of the centroid position.

    .. warning::

       The Voronoi Tessellation step is expensive and quickly becomes
       intractable with dimensions as low as 10 even for a sample
       of size as low as 1000.

    .. versionadded:: 1.9.0

    References
    ----------
    .. [1] Lloyd. "Least Squares Quantization in PCM".
       IEEE Transactions on Information Theory, 1982.
    .. [2] Max J. "Quantizing for minimum distortion".
       IEEE Transactions on Information Theory, 1960.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial import distance
    >>> from scipy.stats._qmc import _lloyd_centroidal_voronoi_tessellation
    >>> rng = np.random.default_rng()
    >>> sample = rng.random((128, 2))

    .. note::

        The samples need to be in :math:`[0, 1]^d`. `scipy.stats.qmc.scale`
        can be used to scale the samples from their
        original bounds to :math:`[0, 1]^d`. And back to their original bounds.

    Compute the quality of the sample using the L1 criterion.

    >>> def l1_norm(sample):
    ...    return distance.pdist(sample, 'cityblock').min()

    >>> l1_norm(sample)
    0.00161...  # random

    Now process the sample using Lloyd's algorithm and check the improvement
    on the L1. The value should increase.

    >>> sample = _lloyd_centroidal_voronoi_tessellation(sample)
    >>> l1_norm(sample)
    0.0278...  # random

    """
    del kwargs
    sample = np.asarray(sample).copy()
    if not sample.ndim == 2:
        raise ValueError('`sample` is not a 2D array')
    if not sample.shape[1] >= 2:
        raise ValueError('`sample` dimension is not >= 2')
    if sample.max() > 1.0 or sample.min() < 0.0:
        raise ValueError('`sample` is not in unit hypercube')
    if qhull_options is None:
        qhull_options = 'Qbb Qc Qz QJ'
        if sample.shape[1] >= 5:
            qhull_options += ' Qx'
    root = -maxiter / np.log(0.1)
    decay = [np.exp(-x / root) + 0.9 for x in range(maxiter)]
    l1_old = _l1_norm(sample=sample)
    for i in range(maxiter):
        sample = _lloyd_iteration(sample=sample, decay=decay[i], qhull_options=qhull_options)
        l1_new = _l1_norm(sample=sample)
        if abs(l1_new - l1_old) < tol:
            break
        else:
            l1_old = l1_new
    return sample