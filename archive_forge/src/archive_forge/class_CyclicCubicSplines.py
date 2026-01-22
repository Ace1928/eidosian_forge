from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
class CyclicCubicSplines(AdditiveGamSmoother):
    """additive smooth components using cyclic cubic regression splines

    This spline basis is the same as in patsy.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        underlying explanatory variable for smooth terms.
        If 2-dimensional, then observations should be in rows and
        explanatory variables in columns.
    df :  int
        numer of basis functions or degrees of freedom
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
    variable_names : {list[str], None}
        The names for the underlying explanatory variables, x used in for
        creating the column and parameter names for the basis functions.
        If ``x`` is a pandas object, then the names will be taken from it.
    """

    def __init__(self, x, df, constraints=None, variable_names=None):
        self.dfs = df
        self.constraints = constraints
        super().__init__(x, variable_names=variable_names)

    def _make_smoothers_list(self):
        smoothers = []
        for v in range(self.k_variables):
            uv_smoother = UnivariateCubicCyclicSplines(self.x[:, v], df=self.dfs[v], constraints=self.constraints, variable_name=self.variable_names[v])
            smoothers.append(uv_smoother)
        return smoothers