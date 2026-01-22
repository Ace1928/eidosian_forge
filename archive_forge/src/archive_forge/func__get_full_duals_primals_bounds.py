from abc import ABCMeta, abstractmethod
from pyomo.contrib.pynumero.interfaces import pyomo_nlp, ampl_nlp
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
import scipy.sparse
from pyomo.common.timing import HierarchicalTimer
def _get_full_duals_primals_bounds(self):
    full_duals_primals_lb = None
    full_duals_primals_ub = None
    if hasattr(self._nlp, 'pyomo_model') and hasattr(self._nlp, 'get_pyomo_variables'):
        pyomo_model = self._nlp.pyomo_model()
        pyomo_variables = self._nlp.get_pyomo_variables()
        if hasattr(pyomo_model, 'ipopt_zL_out'):
            zL_suffix = pyomo_model.ipopt_zL_out
            full_duals_primals_lb = np.empty(self._nlp.n_primals())
            for i, v in enumerate(pyomo_variables):
                if v in zL_suffix:
                    full_duals_primals_lb[i] = zL_suffix[v]
        if hasattr(pyomo_model, 'ipopt_zU_out'):
            zU_suffix = pyomo_model.ipopt_zU_out
            full_duals_primals_ub = np.empty(self._nlp.n_primals())
            for i, v in enumerate(pyomo_variables):
                if v in zU_suffix:
                    full_duals_primals_ub[i] = zU_suffix[v]
    if full_duals_primals_lb is None:
        full_duals_primals_lb = np.ones(self._nlp.n_primals())
    if full_duals_primals_ub is None:
        full_duals_primals_ub = np.ones(self._nlp.n_primals())
    return (full_duals_primals_lb, full_duals_primals_ub)