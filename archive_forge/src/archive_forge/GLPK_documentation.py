import logging
import re
import sys
import csv
import subprocess
from pyomo.common.tempfiles import TempfileManager
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.opt import (
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP

        Process a basic solution
        