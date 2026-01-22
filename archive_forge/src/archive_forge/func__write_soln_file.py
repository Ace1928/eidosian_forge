import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.kernel.block import IBlock
from pyomo.core import Var
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP
def _write_soln_file(self, instance, filename):
    if isinstance(instance, IBlock):
        smap = getattr(instance, '._symbol_maps')[self._smap_id]
    else:
        smap = instance.solutions.symbol_map[self._smap_id]
    byObject = smap.byObject
    column_index = 0
    with open(filename, 'w') as solnfile:
        for var in instance.component_data_objects(Var):
            if var.value and (var.is_integer() or var.is_binary()) and (id(var) in byObject):
                name = byObject[id(var)]
                solnfile.write('{} {} {}\n'.format(column_index, name, var.value))
                column_index += 1