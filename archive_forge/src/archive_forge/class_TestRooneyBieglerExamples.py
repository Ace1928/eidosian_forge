import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
@unittest.skipIf(not parmest.parmest_available, 'Cannot test parmest: required dependencies are missing')
@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
class TestRooneyBieglerExamples(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_model(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import rooney_biegler
        rooney_biegler.main()

    def test_model_with_constraint(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import rooney_biegler_with_constraint
        rooney_biegler_with_constraint.main()

    @unittest.skipUnless(seaborn_available, 'test requires seaborn')
    def test_parameter_estimation_example(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import parameter_estimation_example
        parameter_estimation_example.main()

    @unittest.skipUnless(seaborn_available, 'test requires seaborn')
    def test_bootstrap_example(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import bootstrap_example
        bootstrap_example.main()

    @unittest.skipUnless(seaborn_available, 'test requires seaborn')
    def test_likelihood_ratio_example(self):
        from pyomo.contrib.parmest.examples.rooney_biegler import likelihood_ratio_example
        likelihood_ratio_example.main()