import numpy as np
from scipy import integrate, stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.validation import array_like, float_like
from . import bandwidths
from .kdetools import forrt, revrt, silverman_transform
from .linbin import fast_linbin

        Evaluate density at a point or points.

        Parameters
        ----------
        point : {float, ndarray}
            Point(s) at which to evaluate the density.
        