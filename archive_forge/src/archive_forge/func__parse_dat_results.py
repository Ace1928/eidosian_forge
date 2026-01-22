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
def _parse_dat_results(self, results_filename, statresults_filename):
    with open(statresults_filename, 'r') as statresults_file:
        statresults_text = statresults_file.read()
    stat_vars = dict()
    for line in statresults_text.splitlines()[1:]:
        items = line.split()
        try:
            stat_vars[items[0]] = float(items[1])
        except ValueError:
            stat_vars[items[0]] = float('nan')
    with open(results_filename, 'r') as results_file:
        results_text = results_file.read()
    model_soln = dict()
    for line in results_text.splitlines()[1:]:
        items = line.split()
        model_soln[items[0]] = (items[1], items[2])
    return (model_soln, stat_vars)