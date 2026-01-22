import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections import ComponentMap, Bunch
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver, BranchDirection
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import ILMLicensedSystemCallSolver
from pyomo.solvers.mockmip import MockMIP
from pyomo.core.base import Var, Suffix, active_export_suffix_generator
from pyomo.core.kernel.suffix import export_suffix_generator
from pyomo.core.kernel.block import IBlock
from pyomo.util.components import iter_component
def _validate_file_name(cplex, filename, description):
    """Validate filenames against the set of allowable chaacters in CPLEX.

    Returns the filename, possibly enclosed in double-quotes, or raises
    a ValueError is unallowable characters are found.

    """
    if filename is None:
        return filename
    matches = _validate_file_name.illegal_characters.search(filename)
    if matches:
        raise ValueError('Unallowed character (%s) found in CPLEX %s file path/name.\n\tFor portability reasons, only [%s] are allowed.' % (matches.group(), description, _validate_file_name.allowed_characters.replace('\\', '')))
    if ' ' in filename:
        if cplex.version()[:2] >= (12, 8):
            filename = '"' + filename + '"'
        else:
            raise ValueError('Space detected in CPLEX %s file path/name\n\t%s\nand CPLEX older than version 12.8.  Please either upgrade CPLEX or remove the space from the %s path.' % (description, filename, description))
    return filename