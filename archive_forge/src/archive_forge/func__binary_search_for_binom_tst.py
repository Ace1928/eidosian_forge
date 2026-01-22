from math import sqrt
import numpy as np
from scipy._lib._util import _validate_int
from scipy.optimize import brentq
from scipy.special import ndtri
from ._discrete_distns import binom
from ._common import ConfidenceInterval
def _binary_search_for_binom_tst(a, d, lo, hi):
    """
    Conducts an implicit binary search on a function specified by `a`.

    Meant to be used on the binomial PMF for the case of two-sided tests
    to obtain the value on the other side of the mode where the tail
    probability should be computed. The values on either side of
    the mode are always in order, meaning binary search is applicable.

    Parameters
    ----------
    a : callable
      The function over which to perform binary search. Its values
      for inputs lo and hi should be in ascending order.
    d : float
      The value to search.
    lo : int
      The lower end of range to search.
    hi : int
      The higher end of the range to search.

    Returns
    -------
    int
      The index, i between lo and hi
      such that a(i)<=d<a(i+1)
    """
    while lo < hi:
        mid = lo + (hi - lo) // 2
        midval = a(mid)
        if midval < d:
            lo = mid + 1
        elif midval > d:
            hi = mid - 1
        else:
            return mid
    if a(lo) <= d:
        return lo
    else:
        return lo - 1