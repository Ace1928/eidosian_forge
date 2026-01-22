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
class Sobol(QMCEngine):
    """Engine for generating (scrambled) Sobol' sequences.

    Sobol' sequences are low-discrepancy, quasi-random numbers. Points
    can be drawn using two methods:

    * `random_base2`: safely draw :math:`n=2^m` points. This method
      guarantees the balance properties of the sequence.
    * `random`: draw an arbitrary number of points from the
      sequence. See warning below.

    Parameters
    ----------
    d : int
        Dimensionality of the sequence. Max dimensionality is 21201.
    scramble : bool, optional
        If True, use LMS+shift scrambling. Otherwise, no scrambling is done.
        Default is True.
    bits : int, optional
        Number of bits of the generator. Control the maximum number of points
        that can be generated, which is ``2**bits``. Maximal value is 64.
        It does not correspond to the return type, which is always
        ``np.float64`` to prevent points from repeating themselves.
        Default is None, which for backward compatibility, corresponds to 30.

        .. versionadded:: 1.9.0
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
    Sobol' sequences [1]_ provide :math:`n=2^m` low discrepancy points in
    :math:`[0,1)^{d}`. Scrambling them [3]_ makes them suitable for singular
    integrands, provides a means of error estimation, and can improve their
    rate of convergence. The scrambling strategy which is implemented is a
    (left) linear matrix scramble (LMS) followed by a digital random shift
    (LMS+shift) [2]_.

    There are many versions of Sobol' sequences depending on their
    'direction numbers'. This code uses direction numbers from [4]_. Hence,
    the maximum number of dimension is 21201. The direction numbers have been
    precomputed with search criterion 6 and can be retrieved at
    https://web.maths.unsw.edu.au/~fkuo/sobol/.

    .. warning::

       Sobol' sequences are a quadrature rule and they lose their balance
       properties if one uses a sample size that is not a power of 2, or skips
       the first point, or thins the sequence [5]_.

       If :math:`n=2^m` points are not enough then one should take :math:`2^M`
       points for :math:`M>m`. When scrambling, the number R of independent
       replicates does not have to be a power of 2.

       Sobol' sequences are generated to some number :math:`B` of bits.
       After :math:`2^B` points have been generated, the sequence would
       repeat. Hence, an error is raised.
       The number of bits can be controlled with the parameter `bits`.

    References
    ----------
    .. [1] I. M. Sobol', "The distribution of points in a cube and the accurate
       evaluation of integrals." Zh. Vychisl. Mat. i Mat. Phys., 7:784-802,
       1967.
    .. [2] J. Matousek, "On the L2-discrepancy for anchored boxes."
       J. of Complexity 14, 527-556, 1998.
    .. [3] Art B. Owen, "Scrambling Sobol and Niederreiter-Xing points."
       Journal of Complexity, 14(4):466-489, December 1998.
    .. [4] S. Joe and F. Y. Kuo, "Constructing sobol sequences with better
       two-dimensional projections." SIAM Journal on Scientific Computing,
       30(5):2635-2654, 2008.
    .. [5] Art B. Owen, "On dropping the first Sobol' point."
       :arxiv:`2008.08051`, 2020.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Sobol'.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.Sobol(d=2, scramble=False)
    >>> sample = sampler.random_base2(m=3)
    >>> sample
    array([[0.   , 0.   ],
           [0.5  , 0.5  ],
           [0.75 , 0.25 ],
           [0.25 , 0.75 ],
           [0.375, 0.375],
           [0.875, 0.875],
           [0.625, 0.125],
           [0.125, 0.625]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.013882107204860938

    To continue an existing design, extra points can be obtained
    by calling again `random_base2`. Alternatively, you can skip some
    points like:

    >>> _ = sampler.reset()
    >>> _ = sampler.fast_forward(4)
    >>> sample_continued = sampler.random_base2(m=2)
    >>> sample_continued
    array([[0.375, 0.375],
           [0.875, 0.875],
           [0.625, 0.125],
           [0.125, 0.625]])

    Finally, samples can be scaled to bounds.

    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample_continued, l_bounds, u_bounds)
    array([[3.75 , 3.125],
           [8.75 , 4.625],
           [6.25 , 2.375],
           [1.25 , 3.875]])

    """
    MAXDIM: ClassVar[int] = _MAXDIM

    def __init__(self, d: IntNumber, *, scramble: bool=True, bits: IntNumber | None=None, seed: SeedType=None, optimization: Literal['random-cd', 'lloyd'] | None=None) -> None:
        self._init_quad = {'d': d, 'scramble': True, 'bits': bits, 'optimization': optimization}
        super().__init__(d=d, optimization=optimization, seed=seed)
        if d > self.MAXDIM:
            raise ValueError(f'Maximum supported dimensionality is {self.MAXDIM}.')
        self.bits = bits
        self.dtype_i: type
        if self.bits is None:
            self.bits = 30
        if self.bits <= 32:
            self.dtype_i = np.uint32
        elif 32 < self.bits <= 64:
            self.dtype_i = np.uint64
        else:
            raise ValueError("Maximum supported 'bits' is 64")
        self.maxn = 2 ** self.bits
        self._sv: np.ndarray = np.zeros((d, self.bits), dtype=self.dtype_i)
        _initialize_v(self._sv, dim=d, bits=self.bits)
        if not scramble:
            self._shift: np.ndarray = np.zeros(d, dtype=self.dtype_i)
        else:
            self._scramble()
        self._quasi = self._shift.copy()
        self._scale = 1.0 / 2 ** self.bits
        self._first_point = (self._quasi * self._scale).reshape(1, -1)
        self._first_point = self._first_point.astype(np.float64)

    def _scramble(self) -> None:
        """Scramble the sequence using LMS+shift."""
        self._shift = np.dot(rng_integers(self.rng, 2, size=(self.d, self.bits), dtype=self.dtype_i), 2 ** np.arange(self.bits, dtype=self.dtype_i))
        ltm = np.tril(rng_integers(self.rng, 2, size=(self.d, self.bits, self.bits), dtype=self.dtype_i))
        _cscramble(dim=self.d, bits=self.bits, ltm=ltm, sv=self._sv)

    def _random(self, n: IntNumber=1, *, workers: IntNumber=1) -> np.ndarray:
        """Draw next point(s) in the Sobol' sequence.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        sample: np.ndarray = np.empty((n, self.d), dtype=np.float64)
        if n == 0:
            return sample
        total_n = self.num_generated + n
        if total_n > self.maxn:
            msg = f'At most 2**{self.bits}={self.maxn} distinct points can be generated. {self.num_generated} points have been previously generated, then: n={self.num_generated}+{n}={total_n}. '
            if self.bits != 64:
                msg += 'Consider increasing `bits`.'
            raise ValueError(msg)
        if self.num_generated == 0:
            if not n & n - 1 == 0:
                warnings.warn("The balance properties of Sobol' points require n to be a power of 2.", stacklevel=2)
            if n == 1:
                sample = self._first_point
            else:
                _draw(n=n - 1, num_gen=self.num_generated, dim=self.d, scale=self._scale, sv=self._sv, quasi=self._quasi, sample=sample)
                sample = np.concatenate([self._first_point, sample])[:n]
        else:
            _draw(n=n, num_gen=self.num_generated - 1, dim=self.d, scale=self._scale, sv=self._sv, quasi=self._quasi, sample=sample)
        return sample

    def random_base2(self, m: IntNumber) -> np.ndarray:
        """Draw point(s) from the Sobol' sequence.

        This function draws :math:`n=2^m` points in the parameter space
        ensuring the balance properties of the sequence.

        Parameters
        ----------
        m : int
            Logarithm in base 2 of the number of samples; i.e., n = 2^m.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        n = 2 ** m
        total_n = self.num_generated + n
        if not total_n & total_n - 1 == 0:
            raise ValueError("The balance properties of Sobol' points require n to be a power of 2. {0} points have been previously generated, then: n={0}+2**{1}={2}. If you still want to do this, the function 'Sobol.random()' can be used.".format(self.num_generated, m, total_n))
        return self.random(n)

    def reset(self) -> Sobol:
        """Reset the engine to base state.

        Returns
        -------
        engine : Sobol
            Engine reset to its base state.

        """
        super().reset()
        self._quasi = self._shift.copy()
        return self

    def fast_forward(self, n: IntNumber) -> Sobol:
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : Sobol
            The fast-forwarded engine.

        """
        if self.num_generated == 0:
            _fast_forward(n=n - 1, num_gen=self.num_generated, dim=self.d, sv=self._sv, quasi=self._quasi)
        else:
            _fast_forward(n=n, num_gen=self.num_generated - 1, dim=self.d, sv=self._sv, quasi=self._quasi)
        self.num_generated += n
        return self