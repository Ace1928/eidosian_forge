from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def _invalidate_primals_cache(self):
    self._objective_is_cached = False
    self._grad_objective_is_cached = False
    self._con_full_is_cached = False
    self._jac_full_is_cached = False
    self._hessian_lag_is_cached = False