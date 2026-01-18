import operator
import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
from . import _specfun
from ._comb import _comb_int
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def ynp_zeros(n, nt):
    """Compute zeros of integer-order Bessel function derivatives Yn'(x).

    Compute `nt` zeros of the functions :math:`Y_n'(x)` on the
    interval :math:`(0, \\infty)`. The zeros are returned in ascending
    order.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `nt` zeros of the Bessel derivative function.


    See Also
    --------
    yvp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four roots of the first derivative of the
    Bessel function of second kind for order 0 :math:`Y_0'`.

    >>> from scipy.special import ynp_zeros
    >>> ynp_zeros(0, 4)
    array([ 2.19714133,  5.42968104,  8.59600587, 11.74915483])

    Plot :math:`Y_0`, :math:`Y_0'` and confirm visually that the roots of
    :math:`Y_0'` are located at local extrema of :math:`Y_0`.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import yn, ynp_zeros, yvp
    >>> zeros = ynp_zeros(0, 4)
    >>> xmax = 13
    >>> x = np.linspace(0, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, yn(0, x), label=r'$Y_0$')
    >>> ax.plot(x, yvp(0, x, 1), label=r"$Y_0'$")
    >>> ax.scatter(zeros, np.zeros((4, )), s=30, c='r',
    ...            label=r"Roots of $Y_0'$", zorder=5)
    >>> for root in zeros:
    ...     y0_extremum =  yn(0, root)
    ...     lower = min(0, y0_extremum)
    ...     upper = max(0, y0_extremum)
    ...     ax.vlines(root, lower, upper, color='r')
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.set_ylim(-0.6, 0.6)
    >>> ax.set_xlim(0, xmax)
    >>> plt.legend()
    >>> plt.show()
    """
    return jnyn_zeros(n, nt)[3]