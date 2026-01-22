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
class QMCEngine(ABC):
    """A generic Quasi-Monte Carlo sampler class meant for subclassing.

    QMCEngine is a base class to construct a specific Quasi-Monte Carlo
    sampler. It cannot be used directly as a sampler.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` instance, then the provided
        instance is used.

    Notes
    -----
    By convention samples are distributed over the half-open interval
    ``[0, 1)``. Instances of the class can access the attributes: ``d`` for
    the dimension; and ``rng`` for the random number generator (used for the
    ``seed``).

    **Subclassing**

    When subclassing `QMCEngine` to create a new sampler,  ``__init__`` and
    ``random`` must be redefined.

    * ``__init__(d, seed=None)``: at least fix the dimension. If the sampler
      does not take advantage of a ``seed`` (deterministic methods like
      Halton), this parameter can be omitted.
    * ``_random(n, *, workers=1)``: draw ``n`` from the engine. ``workers``
      is used for parallelism. See `Halton` for example.

    Optionally, two other methods can be overwritten by subclasses:

    * ``reset``: Reset the engine to its original state.
    * ``fast_forward``: If the sequence is deterministic (like Halton
      sequence), then ``fast_forward(n)`` is skipping the ``n`` first draw.

    Examples
    --------
    To create a random sampler based on ``np.random.random``, we would do the
    following:

    >>> from scipy.stats import qmc
    >>> class RandomEngine(qmc.QMCEngine):
    ...     def __init__(self, d, seed=None):
    ...         super().__init__(d=d, seed=seed)
    ...
    ...
    ...     def _random(self, n=1, *, workers=1):
    ...         return self.rng.random((n, self.d))
    ...
    ...
    ...     def reset(self):
    ...         super().__init__(d=self.d, seed=self.rng_seed)
    ...         return self
    ...
    ...
    ...     def fast_forward(self, n):
    ...         self.random(n)
    ...         return self

    After subclassing `QMCEngine` to define the sampling strategy we want to
    use, we can create an instance to sample from.

    >>> engine = RandomEngine(2)
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    We can also reset the state of the generator and resample again.

    >>> _ = engine.reset()
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    """

    @abstractmethod
    def __init__(self, d: IntNumber, *, optimization: Literal['random-cd', 'lloyd'] | None=None, seed: SeedType=None) -> None:
        if not np.issubdtype(type(d), np.integer) or d < 0:
            raise ValueError('d must be a non-negative integer value')
        self.d = d
        if isinstance(seed, np.random.Generator):
            self.rng = _rng_spawn(seed, 1)[0]
        else:
            self.rng = check_random_state(seed)
        self.rng_seed = copy.deepcopy(self.rng)
        self.num_generated = 0
        config = {'n_nochange': 100, 'n_iters': 10000, 'rng': self.rng, 'tol': 1e-05, 'maxiter': 10, 'qhull_options': None}
        self.optimization_method = _select_optimizer(optimization, config)

    @abstractmethod
    def _random(self, n: IntNumber=1, *, workers: IntNumber=1) -> np.ndarray:
        ...

    def random(self, n: IntNumber=1, *, workers: IntNumber=1) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        workers : int, optional
            Only supported with `Halton`.
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Default is 1. It becomes faster
            than one worker for `n` greater than :math:`10^3`.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        sample = self._random(n, workers=workers)
        if self.optimization_method is not None:
            sample = self.optimization_method(sample)
        self.num_generated += n
        return sample

    def integers(self, l_bounds: npt.ArrayLike, *, u_bounds: npt.ArrayLike | None=None, n: IntNumber=1, endpoint: bool=False, workers: IntNumber=1) -> np.ndarray:
        """
        Draw `n` integers from `l_bounds` (inclusive) to `u_bounds`
        (exclusive), or if endpoint=True, `l_bounds` (inclusive) to
        `u_bounds` (inclusive).

        Parameters
        ----------
        l_bounds : int or array-like of ints
            Lowest (signed) integers to be drawn (unless ``u_bounds=None``,
            in which case this parameter is 0 and this value is used for
            `u_bounds`).
        u_bounds : int or array-like of ints, optional
            If provided, one above the largest (signed) integer to be drawn
            (see above for behavior if ``u_bounds=None``).
            If array-like, must contain integer values.
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        endpoint : bool, optional
            If true, sample from the interval ``[l_bounds, u_bounds]`` instead
            of the default ``[l_bounds, u_bounds)``. Defaults is False.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Only supported when using `Halton`
            Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        Notes
        -----
        It is safe to just use the same ``[0, 1)`` to integer mapping
        with QMC that you would use with MC. You still get unbiasedness,
        a strong law of large numbers, an asymptotically infinite variance
        reduction and a finite sample variance bound.

        To convert a sample from :math:`[0, 1)` to :math:`[a, b), b>a`,
        with :math:`a` the lower bounds and :math:`b` the upper bounds,
        the following transformation is used:

        .. math::

            \\text{floor}((b - a) \\cdot \\text{sample} + a)

        """
        if u_bounds is None:
            u_bounds = l_bounds
            l_bounds = 0
        u_bounds = np.atleast_1d(u_bounds)
        l_bounds = np.atleast_1d(l_bounds)
        if endpoint:
            u_bounds = u_bounds + 1
        if not np.issubdtype(l_bounds.dtype, np.integer) or not np.issubdtype(u_bounds.dtype, np.integer):
            message = "'u_bounds' and 'l_bounds' must be integers or array-like of integers"
            raise ValueError(message)
        if isinstance(self, Halton):
            sample = self.random(n=n, workers=workers)
        else:
            sample = self.random(n=n)
        sample = scale(sample, l_bounds=l_bounds, u_bounds=u_bounds)
        sample = np.floor(sample).astype(np.int64)
        return sample

    def reset(self) -> QMCEngine:
        """Reset the engine to base state.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        seed = copy.deepcopy(self.rng_seed)
        self.rng = check_random_state(seed)
        self.num_generated = 0
        return self

    def fast_forward(self, n: IntNumber) -> QMCEngine:
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        self.random(n=n)
        return self