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
@staticmethod
def _options_string_to_dict(istr):
    ans = {}
    istr = istr.strip()
    if not istr:
        return ans
    if istr[0] == "'" or istr[0] == '"':
        istr = eval(istr)
    tokens = shlex.split(istr)
    for token in tokens:
        index = token.find('=')
        if index == -1:
            raise ValueError("Solver options must have the form option=value: '%s'" % istr)
        try:
            val = eval(token[index + 1:])
        except:
            val = token[index + 1:]
        ans[token[:index]] = val
    return ans