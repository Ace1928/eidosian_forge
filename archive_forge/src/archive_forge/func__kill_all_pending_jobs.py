import logging
import os
import re
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.opt import SolverFactory, SolverManagerFactory, OptSolver
from pyomo.opt.parallel.manager import ActionManagerError, ActionStatus
from pyomo.opt.parallel.async_solver import AsynchronousSolverManager
from pyomo.core.base import Block
import pyomo.neos.kestrel
def _kill_all_pending_jobs(self):
    for ah in self._ah.values():
        self.kestrel.kill(ah.job, ah.password)