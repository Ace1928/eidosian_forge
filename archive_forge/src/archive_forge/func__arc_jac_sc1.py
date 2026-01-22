import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def _arc_jac_sc1(w, m):
    """Real inverse Jacobian sc, with complementary modulus

    Solve for z in w = sc(z, 1-m)

    w - real scalar

    m - modulus

    Using that sc(z, m) = -i * sn(i * z, 1 - m)
    cf scipy/signal/_filter_design.py analog for an explanation
    and a reference.

    """
    zcomplex = _arc_jac_sn(1j * w, m)
    if abs(zcomplex.real) > 1e-14:
        raise ValueError
    return zcomplex.imag