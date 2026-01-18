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
@unittest.skipIf(not opt.available(False), 'ipopt_sens is not available')
def test_parameter_example(self):
    d = param_ex.run_example()
    d_correct = {'eta1': 4.5, 'eta2': 1.0, 'x1_init': 0.15, 'x2_init': 0.15, 'x3_init': 0.0, 'cost_sln': 0.5, 'x1_sln': 0.5, 'x2_sln': 0.5, 'x3_sln': 0.0, 'eta1_pert': 4.0, 'eta2_pert': 1.0, 'x1_pert': 0.3333333, 'x2_pert': 0.6666667, 'x3_pert': 0.0, 'cost_pert': 0.55555556}
    for k in d_correct.keys():
        self.assertAlmostEqual(d[k], d_correct[k], 3)