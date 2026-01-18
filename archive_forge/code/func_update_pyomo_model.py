from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.dependencies import attempt_import, numpy as np
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.scc_solver import (
def update_pyomo_model(self):
    for i, var in enumerate(self.get_variables() + self.get_parameters()):
        var.set_value(self._global_values[i], skip_validation=True)