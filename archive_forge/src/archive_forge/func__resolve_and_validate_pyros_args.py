import logging
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.collections import Bunch
from pyomo.core.base.block import Block
from pyomo.core.expr import value
from pyomo.core.base.var import Var
from pyomo.core.base.objective import Objective
from pyomo.contrib.pyros.util import time_code
from pyomo.common.modeling import unique_component_name
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.config import pyros_config
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.solve_data import ROSolveResults
from pyomo.contrib.pyros.pyros_algorithm_methods import ROSolver_iterative_solve
from pyomo.core.base import Constraint
from datetime import datetime
def _resolve_and_validate_pyros_args(self, model, **kwds):
    """
        Resolve and validate arguments to ``self.solve()``.

        Parameters
        ----------
        model : ConcreteModel
            Deterministic model object passed to ``self.solve()``.
        **kwds : dict
            All other arguments to ``self.solve()``.

        Returns
        -------
        config : ConfigDict
            Standardized arguments.

        Note
        ----
        This method can be broken down into three steps:

        1. Cast arguments to ConfigDict. Argument-wise
           validation is performed automatically.
           Note that arguments specified directly take
           precedence over arguments specified indirectly
           through direct argument 'options'.
        2. Inter-argument validation.
        """
    config = self.CONFIG(kwds.pop('options', {}))
    config = config(kwds)
    state_vars = validate_pyros_inputs(model, config)
    return (config, state_vars)