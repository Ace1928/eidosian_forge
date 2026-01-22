from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
def _fix_design(self, m, design_val, fix_opt=True, optimize_option=None):
    """
        Fix design variable

        Parameters
        ----------
        m: model
        design_val: design variable values dict
        fix_opt: if True, fix. Else, unfix
        optimize: a dictionary, keys are design variable name, values are True or False, deciding if this design variable is optimized as DOF this time

        Returns
        -------
        m: model
        """
    for name in self.design_name:
        cuid = pyo.ComponentUID(name)
        var = cuid.find_component_on(m)
        if fix_opt:
            var.fix(design_val[name])
        elif optimize_option is None:
            var.unfix()
        elif optimize_option[name]:
            var.unfix()
    return m