import os
import pyomo.common.unittest as unittest
import pyomo.opt
import pyomo.solvers.plugins.solvers
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC

        Verify that options can be passed in.
        