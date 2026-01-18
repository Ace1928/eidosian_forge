import os
import sys
import time
import logging
import subprocess
from io import StringIO
from contextlib import nullcontext
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.log import is_debug_set, LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import TeeStream
import pyomo.common
from pyomo.opt.base import ResultsFormat
from pyomo.opt.base.solvers import OptSolver
from pyomo.opt.results import SolverStatus, SolverResults
Returns the default results format for different problem
        formats.
        