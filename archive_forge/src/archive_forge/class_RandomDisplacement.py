import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
class RandomDisplacement:
    """Add a random displacement of maximum size `stepsize` to each coordinate.

    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    """

    def __init__(self, stepsize=0.5, random_gen=None):
        self.stepsize = stepsize
        self.random_gen = check_random_state(random_gen)

    def __call__(self, x):
        x += self.random_gen.uniform(-self.stepsize, self.stepsize, np.shape(x))
        return x