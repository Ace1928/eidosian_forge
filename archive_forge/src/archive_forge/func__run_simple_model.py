from io import StringIO
import shlex
from tempfile import mkdtemp
import os, sys, math, logging, shutil, time, subprocess
from pyomo.core.base import Constraint, Var, value, Objective
from pyomo.opt import ProblemFormat, SolverFactory
import pyomo.common
from pyomo.common.collections import Bunch
from pyomo.common.tee import TeeStream
from pyomo.opt.base.solvers import _extract_version
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.objective import IObjective
from pyomo.core.kernel.variable import IVariable
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.opt.results import (
from pyomo.common.dependencies import attempt_import
def _run_simple_model(self, n):
    solver_exec = self.executable()
    if solver_exec is None:
        return False
    tmpdir = mkdtemp()
    try:
        test = os.path.join(tmpdir, 'test.gms')
        with open(test, 'w') as FILE:
            FILE.write(self._simple_model(n))
        result = subprocess.run([solver_exec, test, 'curdir=' + tmpdir, 'lo=0'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return not result.returncode
    finally:
        shutil.rmtree(tmpdir)
    return False