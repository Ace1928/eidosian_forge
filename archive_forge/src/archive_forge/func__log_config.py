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
def _log_config(self, logger, config, exclude_options=None, **log_kwargs):
    """
        Log PyROS solver options.

        Parameters
        ----------
        logger : logging.Logger
            Logger for the solver options.
        config : ConfigDict
            PyROS solver options.
        exclude_options : None or iterable of str, optional
            Options (keys of the ConfigDict) to exclude from
            logging. If `None` passed, then the names of the
            required arguments to ``self.solve()`` are skipped.
        **log_kwargs : dict, optional
            Keyword arguments to each statement of ``logger.log()``.
        """
    if exclude_options is None:
        exclude_options = ['first_stage_variables', 'second_stage_variables', 'uncertain_params', 'uncertainty_set', 'local_solver', 'global_solver']
    logger.log(msg='Solver options:', **log_kwargs)
    for key, val in config.items():
        if key not in exclude_options:
            logger.log(msg=f' {key}={val!r}', **log_kwargs)
    logger.log(msg='-' * self._LOG_LINE_LENGTH, **log_kwargs)