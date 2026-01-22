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
def _nolan_round_x_near_zeta(x0, alpha, zeta, x_tol_near_zeta):
    """Round x close to zeta for Nolan's method in [NO]."""
    if np.abs(x0 - zeta) < x_tol_near_zeta * alpha ** (1 / alpha):
        x0 = zeta
    return x0