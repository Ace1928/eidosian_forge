from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def _collect_nlp_structure(self):
    """
        Collect characteristics of the NLP from the ASL interface
        """
    self._n_primals = self._asl.get_n_vars()
    self._n_con_full = self._asl.get_n_constraints()
    self._nnz_jac_full = self._asl.get_nnz_jac_g()
    self._nnz_hess_lag_lower = self._asl.get_nnz_hessian_lag()
    self._init_primals = np.zeros(self._n_primals, dtype=np.float64)
    self._init_duals_full = np.zeros(self._n_con_full, dtype=np.float64)
    self._asl.get_init_x(self._init_primals)
    self._asl.get_init_multipliers(self._init_duals_full)
    self._init_primals.flags.writeable = False
    self._init_duals_full.flags.writeable = False
    self._primals_lb = np.zeros(self._n_primals, dtype=np.float64)
    self._primals_ub = np.zeros(self._n_primals, dtype=np.float64)
    self._asl.get_x_lower_bounds(self._primals_lb)
    self._asl.get_x_upper_bounds(self._primals_ub)
    self._primals_lb.flags.writeable = False
    self._primals_ub.flags.writeable = False
    self._con_full_lb = np.zeros(self._n_con_full, dtype=np.float64)
    self._con_full_ub = np.zeros(self._n_con_full, dtype=np.float64)
    self._asl.get_g_lower_bounds(self._con_full_lb)
    self._asl.get_g_upper_bounds(self._con_full_ub)
    bounds_difference = self._primals_ub - self._primals_lb
    if np.any(bounds_difference < 0):
        print(np.where(bounds_difference < 0))
        raise RuntimeError('Some variables have lower bounds that are greater than the upper bounds.')
    self._build_constraint_maps()
    self._con_ineq_lb = np.compress(self._con_full_ineq_mask, self._con_full_lb)
    self._con_ineq_ub = np.compress(self._con_full_ineq_mask, self._con_full_ub)
    self._con_ineq_lb.flags.writeable = False
    self._con_ineq_ub.flags.writeable = False
    self._init_duals_eq = np.compress(self._con_full_eq_mask, self._init_duals_full)
    self._init_duals_ineq = np.compress(self._con_full_ineq_mask, self._init_duals_full)
    self._init_duals_eq.flags.writeable = False
    self._init_duals_ineq.flags.writeable = False
    self._con_full_rhs = self._con_full_ub.copy()
    self._con_full_rhs[~self._con_full_eq_mask] = 0.0
    self._con_full_lb[self._con_full_eq_mask] = 0.0
    self._con_full_ub[self._con_full_eq_mask] = 0.0
    self._con_full_lb.flags.writeable = False
    self._con_full_ub.flags.writeable = False
    self._n_con_eq = len(self._con_eq_full_map)
    self._n_con_ineq = len(self._con_ineq_full_map)
    self._irows_jac_full = np.zeros(self._nnz_jac_full, dtype=np.intc)
    self._jcols_jac_full = np.zeros(self._nnz_jac_full, dtype=np.intc)
    self._asl.struct_jac_g(self._irows_jac_full, self._jcols_jac_full)
    self._irows_jac_full -= 1
    self._jcols_jac_full -= 1
    self._irows_jac_full.flags.writeable = False
    self._jcols_jac_full.flags.writeable = False
    self._nz_con_full_eq_mask = np.isin(self._irows_jac_full, self._con_eq_full_map)
    self._nz_con_full_ineq_mask = np.logical_not(self._nz_con_full_eq_mask)
    self._irows_jac_eq = np.compress(self._nz_con_full_eq_mask, self._irows_jac_full)
    self._jcols_jac_eq = np.compress(self._nz_con_full_eq_mask, self._jcols_jac_full)
    self._irows_jac_ineq = np.compress(self._nz_con_full_ineq_mask, self._irows_jac_full)
    self._jcols_jac_ineq = np.compress(self._nz_con_full_ineq_mask, self._jcols_jac_full)
    self._nnz_jac_eq = len(self._irows_jac_eq)
    self._nnz_jac_ineq = len(self._irows_jac_ineq)
    self._con_full_eq_map = full_eq_map = {self._con_eq_full_map[i]: i for i in range(self._n_con_eq)}
    for i, v in enumerate(self._irows_jac_eq):
        self._irows_jac_eq[i] = full_eq_map[v]
    self._con_full_ineq_map = full_ineq_map = {self._con_ineq_full_map[i]: i for i in range(self._n_con_ineq)}
    for i, v in enumerate(self._irows_jac_ineq):
        self._irows_jac_ineq[i] = full_ineq_map[v]
    self._irows_jac_eq.flags.writeable = False
    self._jcols_jac_eq.flags.writeable = False
    self._irows_jac_ineq.flags.writeable = False
    self._jcols_jac_ineq.flags.writeable = False
    self._nnz_jac_eq = len(self._jcols_jac_eq)
    self._nnz_jac_ineq = len(self._jcols_jac_ineq)
    self._irows_hess = np.zeros(self._nnz_hess_lag_lower, dtype=np.intc)
    self._jcols_hess = np.zeros(self._nnz_hess_lag_lower, dtype=np.intc)
    self._asl.struct_hes_lag(self._irows_hess, self._jcols_hess)
    self._irows_hess -= 1
    self._jcols_hess -= 1
    diff = self._irows_hess - self._jcols_hess
    self._lower_hess_mask = np.where(diff != 0)
    lower = self._lower_hess_mask
    self._irows_hess = np.concatenate((self._irows_hess, self._jcols_hess[lower]))
    self._jcols_hess = np.concatenate((self._jcols_hess, self._irows_hess[lower]))
    self._nnz_hessian_lag = self._irows_hess.size
    self._irows_hess.flags.writeable = False
    self._jcols_hess.flags.writeable = False