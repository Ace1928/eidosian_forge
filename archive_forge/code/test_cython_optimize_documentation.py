import numpy.testing as npt
from scipy.optimize.cython_optimize import _zeros

Test Cython optimize zeros API functions: ``bisect``, ``ridder``, ``brenth``,
and ``brentq`` in `scipy.optimize.cython_optimize`, by finding the roots of a
3rd order polynomial given a sequence of constant terms, ``a0``, and fixed 1st,
2nd, and 3rd order terms in ``args``.

.. math::

    f(x, a0, args) =  ((args[2]*x + args[1])*x + args[0])*x + a0

The 3rd order polynomial function is written in Cython and called in a Python
wrapper named after the zero function. See the private ``_zeros`` Cython module
in `scipy.optimize.cython_optimze` for more information.
