import logging
import re
import sys
import itertools
import operator
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.common.dependencies import attempt_import
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import is_fixed, value, minimize, maximize
from pyomo.core.base.suffix import Suffix
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt.base.solvers import OptSolver
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.common.collections import ComponentMap, ComponentSet, Bunch
from pyomo.opt import SolverFactory
from pyomo.core.kernel.conic import (
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
def _apply_solver(self):
    StaleFlagManager.mark_all_as_stale()
    if self._tee:

        def _process_stream(msg):
            sys.stdout.write(msg)
            sys.stdout.flush()
        self._solver_model.set_Stream(mosek.streamtype.log, _process_stream)
    if self._keepfiles:
        logger.info('Solver log file: {}'.format(self._log_file))
    for key, option in self.options.items():
        try:
            param = key.split('.')
            if param[0] == 'mosek':
                param.pop(0)
            param = getattr(mosek, param[0])(param[1])
            if 'sparam' in key.split('.'):
                self._solver_model.putstrparam(param, option)
            elif 'dparam' in key.split('.'):
                self._solver_model.putdouparam(param, option)
            elif 'iparam' in key.split('.'):
                if isinstance(option, str):
                    option = option.split('.')
                    if option[0] == 'mosek':
                        option.pop('mosek')
                    option = getattr(mosek, option[0])(option[1])
                else:
                    self._solver_model.putintparam(param, option)
        except (TypeError, AttributeError):
            raise
    try:
        self._termcode = self._solver_model.optimize()
        self._solver_model.solutionsummary(mosek.streamtype.msg)
    except mosek.Error as e:
        self._mosek_env.checkinall()
        logger.error(e)
        raise
    return Bunch(rc=None, log=None)