from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_rho(self, rho_new):
    """
        Update set-size parameter rho
        """
    if rho_new <= 0:
        raise ValueError('rho must be positive')
    self.work.settings.rho = np.minimum(np.maximum(rho_new, RHO_MIN), RHO_MAX)
    ineq_ind = np.where(self.work.constr_type == 0)
    eq_ind = np.where(self.work.constr_type == 1)
    self.work.rho_vec[ineq_ind] = self.work.settings.rho
    self.work.rho_vec[eq_ind] = RHO_EQ_OVER_RHO_INEQ * self.work.settings.rho
    self.work.rho_inv_vec = np.reciprocal(self.work.rho_vec)
    self.work.linsys_solver = linsys_solver(self.work)