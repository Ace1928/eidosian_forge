from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
def ComputeSecondStageCost_rule(m):
    return sum(((m.ca[t] - ca_meas[t]) ** 2 + (m.cb[t] - cb_meas[t]) ** 2 + (m.cc[t] - cc_meas[t]) ** 2 for t in meas_t))