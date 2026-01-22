from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _smooth_basis_for_single_variable(self):
    basis = dmatrix('cc(x, df=' + str(self.df) + ') - 1', {'x': self.x})
    self.design_info = basis.design_info
    n_inner_knots = self.df - 2 + 1
    all_knots = _get_all_sorted_knots(self.x, n_inner_knots=n_inner_knots, inner_knots=None, lower_bound=None, upper_bound=None)
    b, d = self._get_b_and_d(all_knots)
    s = self._get_s(b, d)
    return (basis, None, None, s)