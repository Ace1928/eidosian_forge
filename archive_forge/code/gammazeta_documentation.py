import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\

    This function implements multipurpose evaluation of the gamma
    function, G(x), as well as the following versions of the same:

    type = 0 -- G(x)                    [standard gamma function]
    type = 1 -- G(x+1) = x*G(x+1) = x!  [factorial]
    type = 2 -- 1/G(x)                  [reciprocal gamma function]
    type = 3 -- log(|G(x)|)             [log-gamma function, real part]
    