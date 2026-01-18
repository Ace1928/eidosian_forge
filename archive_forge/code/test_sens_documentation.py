from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Param, Var, Block, Suffix, value
from pyomo.opt import SolverFactory
from pyomo.dae import ContinuousSet
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap
from pyomo.core.expr import identify_variables
from pyomo.contrib.sensitivity_toolbox.sens import sipopt, kaug, sensitivity_calculation
import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_ex
import pyomo.contrib.sensitivity_toolbox.examples.parameter_kaug as param_kaug_ex
import pyomo.contrib.sensitivity_toolbox.examples.feedbackController as fc
import pyomo.contrib.sensitivity_toolbox.examples.rangeInequality as ri
import pyomo.contrib.sensitivity_toolbox.examples.HIV_Transmission as hiv

Unit Tests for interfacing with sIPOPT and k_aug
