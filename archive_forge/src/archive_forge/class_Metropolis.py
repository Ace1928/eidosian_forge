import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state
class Metropolis:
    """Metropolis acceptance criterion.

    Parameters
    ----------
    T : float
        The "temperature" parameter for the accept or reject criterion.
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Random number generator used for acceptance test.

    """

    def __init__(self, T, random_gen=None):
        self.beta = 1.0 / T if T != 0 else float('inf')
        self.random_gen = check_random_state(random_gen)

    def accept_reject(self, res_new, res_old):
        """
        Assuming the local search underlying res_new was successful:
        If new energy is lower than old, it will always be accepted.
        If new is higher than old, there is a chance it will be accepted,
        less likely for larger differences.
        """
        with np.errstate(invalid='ignore'):
            prod = -(res_new.fun - res_old.fun) * self.beta
            w = math.exp(min(0, prod))
        rand = self.random_gen.uniform()
        return w >= rand and (res_new.success or not res_old.success)

    def __call__(self, *, res_new, res_old):
        """
        f_new and f_old are mandatory in kwargs
        """
        return bool(self.accept_reject(res_new, res_old))