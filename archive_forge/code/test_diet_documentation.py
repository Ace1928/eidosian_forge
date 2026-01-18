import json
import os
import pyomo.common.unittest as unittest
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers

        Run Pyomo with the given arguments. `args` should be a list with
        one argument token per string item.
        