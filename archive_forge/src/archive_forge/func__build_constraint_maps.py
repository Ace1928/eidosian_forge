from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def _build_constraint_maps(self):
    """Creates internal maps and masks that convert from the full
        vector of constraints (the vector that includes all equality
        and inequality constraints combined) to separate vectors that
        include the equality and inequality constraints only.
        """
    bounds_difference = self._con_full_ub - self._con_full_lb
    inconsistent_bounds = np.any(bounds_difference < 0.0)
    if inconsistent_bounds:
        raise RuntimeError('Bounds on range constraints found with upper bounds set below the lower bounds.')
    abs_bounds_difference = np.absolute(bounds_difference)
    tolerance_equalities = 1e-08
    self._con_full_eq_mask = abs_bounds_difference < tolerance_equalities
    self._con_eq_full_map = self._con_full_eq_mask.nonzero()[0]
    self._con_full_ineq_mask = abs_bounds_difference >= tolerance_equalities
    self._con_ineq_full_map = self._con_full_ineq_mask.nonzero()[0]
    self._con_full_eq_mask.flags.writeable = False
    self._con_eq_full_map.flags.writeable = False
    self._con_full_ineq_mask.flags.writeable = False
    self._con_ineq_full_map.flags.writeable = False
    '\n        #TODO: Can we simplify this logic?\n        con_full_fulllb_mask = np.isfinite(self._con_full_lb) * self._con_full_ineq_mask + self._con_full_eq_mask\n        con_fulllb_full_map = con_full_fulllb_mask.nonzero()[0]\n        con_full_fullub_mask = np.isfinite(self._con_full_ub) * self._con_full_ineq_mask + self._con_full_eq_mask\n        con_fullub_full_map = con_full_fullub_mask.nonzero()[0]\n\n        self._ineq_lb_mask = np.isin(self._ineq_g_map, lb_g_map)\n        self._lb_ineq_map = np.where(self._ineq_lb_mask)[0]\n        self._ineq_ub_mask = np.isin(self._ineq_g_map, ub_g_map)\n        self._ub_ineq_map = np.where(self._ineq_ub_mask)[0]\n        self._ineq_lb_mask.flags.writeable = False\n        self._lb_ineq_map.flags.writeable = False\n        self._ineq_ub_mask.flags.writeable = False\n        self._ub_ineq_map.flags.writeable = False\n        '