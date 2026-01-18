import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
import pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models as ex_models
def test_scaling_all_missing(self):
    m = self.create_model_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputs())
    m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
    pyomo_nlp = PyomoNLPWithGreyBoxBlocks(m)
    fs = pyomo_nlp.get_obj_scaling()
    xs = pyomo_nlp.get_primals_scaling()
    cs = pyomo_nlp.get_constraints_scaling()
    self.assertIsNone(fs)
    self.assertIsNone(xs)
    self.assertIsNone(cs)