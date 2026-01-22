from pyomo.common.dependencies import numpy as np, numpy_available, pandas_available
import pyomo.common.unittest as unittest
from pyomo.contrib.doe import DesignOfExperiments, MeasurementVariables, DesignVariables
from pyomo.environ import value, ConcreteModel
from pyomo.contrib.doe.examples.reactor_kinetics import create_model, disc_for_measure
from pyomo.opt import SolverFactory
class Test_example_options(unittest.TestCase):
    """Test the three options in the kinetics example."""

    def test_setUP(self):
        mod = create_model(model_option='parmest')
        mod = ConcreteModel()
        create_model(mod, model_option='stage1')
        create_model(mod, model_option='stage2')
        with self.assertRaises(ValueError):
            create_model(model_option='stage1')
        with self.assertRaises(ValueError):
            create_model(model_option='stage2')
        with self.assertRaises(ValueError):
            create_model(model_option='NotDefine')