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
def _pdf_single_value_piecewise_Z0(x0, alpha, beta, **kwds):
    quad_eps = kwds.get('quad_eps', _QUAD_EPS)
    x_tol_near_zeta = kwds.get('piecewise_x_tol_near_zeta', 0.005)
    alpha_tol_near_one = kwds.get('piecewise_alpha_tol_near_one', 0.005)
    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    x0, alpha, beta = _nolan_round_difficult_input(x0, alpha, beta, zeta, x_tol_near_zeta, alpha_tol_near_one)
    if alpha == 2.0:
        return _norm_pdf(x0 / np.sqrt(2)) / np.sqrt(2)
    elif alpha == 0.5 and beta == 1.0:
        _x = x0 + 1
        if _x <= 0:
            return 0
        return 1 / np.sqrt(2 * np.pi * _x) / _x * np.exp(-1 / (2 * _x))
    elif alpha == 0.5 and beta == 0.0 and (x0 != 0):
        S, C = sc.fresnel([1 / np.sqrt(2 * np.pi * np.abs(x0))])
        arg = 1 / (4 * np.abs(x0))
        return (np.sin(arg) * (0.5 - S[0]) + np.cos(arg) * (0.5 - C[0])) / np.sqrt(2 * np.pi * np.abs(x0) ** 3)
    elif alpha == 1.0 and beta == 0.0:
        return 1 / (1 + x0 ** 2) / np.pi
    return _pdf_single_value_piecewise_post_rounding_Z0(x0, alpha, beta, quad_eps, x_tol_near_zeta)