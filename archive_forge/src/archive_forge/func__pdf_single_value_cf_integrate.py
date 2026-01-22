import warnings
from functools import partial
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy.integrate._quadrature import _builtincoeffs
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import scipy.special as sc
from scipy._lib._util import _lazywhere
from .._distn_infrastructure import rv_continuous, _ShapeInfo
from .._continuous_distns import uniform, expon, _norm_pdf, _norm_cdf
from .levyst import Nolan
from scipy._lib.doccer import inherit_docstring_from
def _pdf_single_value_cf_integrate(Phi, x, alpha, beta, **kwds):
    """To improve DNI accuracy convert characteristic function in to real
    valued integral using Euler's formula, then exploit cosine symmetry to
    change limits to [0, inf). Finally use cosine addition formula to split
    into two parts that can be handled by weighted quad pack.
    """
    quad_eps = kwds.get('quad_eps', _QUAD_EPS)

    def integrand1(t):
        if t == 0:
            return 0
        return np.exp(-t ** alpha) * np.cos(beta * t ** alpha * Phi(alpha, t))

    def integrand2(t):
        if t == 0:
            return 0
        return np.exp(-t ** alpha) * np.sin(beta * t ** alpha * Phi(alpha, t))
    with np.errstate(invalid='ignore'):
        int1, *ret1 = integrate.quad(integrand1, 0, np.inf, weight='cos', wvar=x, limit=1000, epsabs=quad_eps, epsrel=quad_eps, full_output=1)
        int2, *ret2 = integrate.quad(integrand2, 0, np.inf, weight='sin', wvar=x, limit=1000, epsabs=quad_eps, epsrel=quad_eps, full_output=1)
    return (int1 + int2) / np.pi