from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,

    Solve a Risch Differential Equation: Dy + f*y == g.

    Explanation
    ===========

    See the outline in the docstring of rde.py for more information
    about the procedure used.  Either raise NonElementaryIntegralException, in
    which case there is no solution y in the given differential field,
    or return y in k(t) satisfying Dy + f*y == g, or raise
    NotImplementedError, in which case, the algorithms necessary to
    solve the given Risch Differential Equation have not yet been
    implemented.
    