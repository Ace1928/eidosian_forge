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
def _log_intro(self, logger, **log_kwargs):
    """
        Log PyROS solver introductory messages.

        Parameters
        ----------
        logger : logging.Logger
            Logger through which to emit messages.
        **log_kwargs : dict, optional
            Keyword arguments to ``logger.log()`` callable.
            Should not include `msg`.
        """
    logger.log(msg='=' * self._LOG_LINE_LENGTH, **log_kwargs)
    logger.log(msg=f'PyROS: The Pyomo Robust Optimization Solver, v{self.version()}.', **log_kwargs)
    version_info = _get_pyomo_version_info()
    version_info_str = ' ' * len('PyROS: ') + ('\n' + ' ' * len('PyROS: ')).join((f'{key}: {val}' for key, val in version_info.items()))
    logger.log(msg=version_info_str, **log_kwargs)
    logger.log(msg=f'{' ' * len('PyROS:')} Invoked at UTC {datetime.utcnow().isoformat()}', **log_kwargs)
    logger.log(msg='', **log_kwargs)
    logger.log(msg='Developed by: Natalie M. Isenberg (1), Jason A. F. Sherman (1),', **log_kwargs)
    logger.log(msg=f'{' ' * len('Developed by:')} John D. Siirola (2), Chrysanthos E. Gounaris (1)', **log_kwargs)
    logger.log(msg='(1) Carnegie Mellon University, Department of Chemical Engineering', **log_kwargs)
    logger.log(msg='(2) Sandia National Laboratories, Center for Computing Research', **log_kwargs)
    logger.log(msg='', **log_kwargs)
    logger.log(msg='The developers gratefully acknowledge support from the U.S. Department', **log_kwargs)
    logger.log(msg="of Energy's Institute for the Design of Advanced Energy Systems (IDAES).", **log_kwargs)
    logger.log(msg='=' * self._LOG_LINE_LENGTH, **log_kwargs)