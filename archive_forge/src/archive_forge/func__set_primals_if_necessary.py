import numpy as np
import abc
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.environ import Var, Constraint, value
from pyomo.core.base.var import _VarData
from pyomo.common.modeling import unique_component_name
def _set_primals_if_necessary(self, primals):
    if not np.array_equal(primals, self._cached_primals):
        self._pyomo_nlp.set_primals(primals)
        ex_inputs = self._ex_io_inputs_from_full_primals(primals)
        self._ex_io_model.set_inputs(ex_inputs)
        self._cached_primals = primals.copy()