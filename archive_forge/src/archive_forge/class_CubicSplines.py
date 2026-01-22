from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
class CubicSplines(AdditiveGamSmoother):
    """additive smooth components using cubic splines as in Wood 2006.

    Note, these splines do NOT use the same spline basis as
    ``Cubic Regression Splines``.
    """

    def __init__(self, x, df, constraints='center', transform='domain', variable_names=None):
        self.dfs = df
        self.constraints = constraints
        self.transform = transform
        super().__init__(x, constraints=constraints, variable_names=variable_names)

    def _make_smoothers_list(self):
        smoothers = []
        for v in range(self.k_variables):
            uv_smoother = UnivariateCubicSplines(self.x[:, v], df=self.dfs[v], constraints=self.constraints, transform=self.transform, variable_name=self.variable_names[v])
            smoothers.append(uv_smoother)
        return smoothers