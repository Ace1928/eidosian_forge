import numpy as np
import numpy.linalg as L
from scipy.linalg import solveh_banded
from scipy.optimize import golden
from models import _hbspline     #removed because this was segfaulting
import warnings

        Fit smoothing spline trying to optimize GCV.

        Try to find a bracketing interval for scipy.optimize.golden
        based on bracket.

        It is probably best to use target_df instead, as it is
        sometimes difficult to find a bracketing interval.

        INPUTS:
           y       -- response variable
           x       -- if None, uses self.x
           df      -- target degrees of freedom
           weights -- optional array of weights
           tol     -- (relative) tolerance for convergence
           brack   -- an initial guess at the bracketing interval

        OUTPUTS: None
           The smoothing spline is determined by self.coef,
           subsequent calls of __call__ will be the smoothing spline.
        