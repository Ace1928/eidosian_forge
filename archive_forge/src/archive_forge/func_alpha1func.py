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
def alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
    return 2 / np.pi * ((np.pi / 2 + bTH) * tanTH - beta * np.log(np.pi / 2 * W * cosTH / (np.pi / 2 + bTH)))