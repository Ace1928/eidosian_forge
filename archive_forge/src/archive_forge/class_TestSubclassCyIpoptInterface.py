import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
class TestSubclassCyIpoptInterface(unittest.TestCase):

    def test_subclass_no_init(self):

        class MyCyIpoptProblem(CyIpoptProblemInterface):

            def __init__(self):
                pass

            def x_init(self):
                pass

            def x_lb(self):
                pass

            def x_ub(self):
                pass

            def g_lb(self):
                pass

            def g_ub(self):
                pass

            def scaling_factors(self):
                pass

            def objective(self, x):
                pass

            def gradient(self, x):
                pass

            def constraints(self, x):
                pass

            def jacobianstructure(self):
                pass

            def jacobian(self, x):
                pass

            def hessianstructure(self):
                pass

            def hessian(self, x, y, obj_factor):
                pass
        problem = MyCyIpoptProblem()
        x0 = []
        msg = '__init__ has not been called'
        with self.assertRaisesRegex(RuntimeError, msg):
            problem.solve(x0)