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
Solve a model.

        Parameters
        ----------
        model: ConcreteModel
            The deterministic model.
        first_stage_variables: VarData, Var, or iterable of VarData/Var
            First-stage model variables (or design variables).
        second_stage_variables: VarData, Var, or iterable of VarData/Var
            Second-stage model variables (or control variables).
        uncertain_params: ParamData, Param, or iterable of ParamData/Param
            Uncertain model parameters.
            The `mutable` attribute for all uncertain parameter objects
            must be set to True.
        uncertainty_set: UncertaintySet
            Uncertainty set against which the solution(s) returned
            will be confirmed to be robust.
        local_solver: str or solver type
            Subordinate local NLP solver.
            If a `str` is passed, then the `str` is cast to
            ``SolverFactory(local_solver)``.
        global_solver: str or solver type
            Subordinate global NLP solver.
            If a `str` is passed, then the `str` is cast to
            ``SolverFactory(global_solver)``.

        Returns
        -------
        return_soln : ROSolveResults
            Summary of PyROS termination outcome.

        