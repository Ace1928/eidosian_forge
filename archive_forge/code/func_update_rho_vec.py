from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_rho_vec(self):
    """
        Update values of rho_vec and refactor if constraints change.
        """
    loose_ind = np.where(np.logical_and(self.work.data.l < -OSQP_INFTY * MIN_SCALING, self.work.data.u > OSQP_INFTY * MIN_SCALING))[0]
    eq_ind = np.where(self.work.data.u - self.work.data.l < RHO_TOL)[0]
    ineq_ind = np.setdiff1d(np.setdiff1d(np.arange(self.work.data.m), loose_ind), eq_ind)
    old_loose_ind = np.where(self.work.constr_type == -1)
    old_eq_ind = np.where(self.work.constr_type == 1)
    old_ineq_ind = np.where(self.work.constr_type == 0)
    constr_type_changed = (loose_ind != old_loose_ind).any() or (eq_ind != old_eq_ind).any() or (ineq_ind != old_ineq_ind).any()
    self.work.constr_type[loose_ind] = -1
    self.work.constr_type[eq_ind] = 1
    self.work.constr_type[ineq_ind] = 0
    self.work.rho_vec[loose_ind] = RHO_MIN
    self.work.rho_vec[eq_ind] = RHO_EQ_OVER_RHO_INEQ * self.work.settings.rho
    self.work.rho_vec[ineq_ind] = self.work.settings.rho
    self.work.rho_inv_vec = np.reciprocal(self.work.rho_vec)
    if constr_type_changed:
        self.work.linsys_solver = linsys_solver(self.work)