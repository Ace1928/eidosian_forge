import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _gen_dV_dPar(self, ex_r, solver, group_ix, max_ix=None):
    """
        A generator that yields the element-wise derivative of the
        marginal covariance matrix with respect to the random effects
        variance and covariance parameters.

        ex_r : array_like
            The random effects design matrix
        solver : function
            A function that given x returns V^{-1}x, where V
            is the group's marginal covariance matrix.
        group_ix : int
            The group index
        max_ix : {int, None}
            If not None, the generator ends when this index
            is reached.
        """
    axr = solver(ex_r)
    jj = 0
    for j1 in range(self.k_re):
        for j2 in range(j1 + 1):
            if max_ix is not None and jj > max_ix:
                return
            mat_l, mat_r = (ex_r[:, j1:j1 + 1], ex_r[:, j2:j2 + 1])
            vsl, vsr = (axr[:, j1:j1 + 1], axr[:, j2:j2 + 1])
            yield (jj, mat_l, mat_r, vsl, vsr, j1 == j2)
            jj += 1
    for j, _ in enumerate(self.exog_vc.names):
        if max_ix is not None and jj > max_ix:
            return
        mat = self.exog_vc.mats[j][group_ix]
        axmat = solver(mat)
        yield (jj, mat, mat, axmat, axmat, True)
        jj += 1