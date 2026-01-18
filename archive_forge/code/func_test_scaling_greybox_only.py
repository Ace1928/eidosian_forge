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
def test_scaling_greybox_only(self):
    m = self.create_model_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputsScaleBoth())
    m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
    pyomo_nlp = PyomoNLPWithGreyBoxBlocks(m)
    comparison_x_order = ['egb.inputs[Pin]', 'egb.inputs[c]', 'egb.inputs[F]', 'egb.inputs[P1]', 'egb.inputs[P3]', 'egb.outputs[P2]', 'egb.outputs[Pout]', 'hin', 'hout']
    x_order = pyomo_nlp.primals_names()
    comparison_c_order = ['egb.pdrop1', 'egb.pdrop3', 'egb.output_constraints[P2]', 'egb.output_constraints[Pout]', 'incon', 'outcon']
    c_order = pyomo_nlp.constraint_names()
    fs = pyomo_nlp.get_obj_scaling()
    self.assertEqual(fs, 1.0)
    xs = pyomo_nlp.get_primals_scaling()
    comparison_xs = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    check_vectors_specific_order(self, xs, x_order, comparison_xs, comparison_x_order)
    cs = pyomo_nlp.get_constraints_scaling()
    comparison_cs = np.asarray([3.1, 3.2, 4.1, 4.2, 1, 1], dtype=np.float64)
    check_vectors_specific_order(self, cs, c_order, comparison_cs, comparison_c_order)
    m = self.create_model_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputsScaleEqualities())
    m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
    pyomo_nlp = PyomoNLPWithGreyBoxBlocks(m)
    cs = pyomo_nlp.get_constraints_scaling()
    comparison_cs = np.asarray([3.1, 3.2, 1, 1, 1, 1], dtype=np.float64)
    check_vectors_specific_order(self, cs, c_order, comparison_cs, comparison_c_order)
    m = self.create_model_two_equalities_two_outputs(ex_models.PressureDropTwoEqualitiesTwoOutputsScaleOutputs())
    m.obj = pyo.Objective(expr=(m.egb.outputs['Pout'] - 20) ** 2)
    pyomo_nlp = PyomoNLPWithGreyBoxBlocks(m)
    cs = pyomo_nlp.get_constraints_scaling()
    comparison_cs = np.asarray([1, 1, 4.1, 4.2, 1, 1], dtype=np.float64)
    check_vectors_specific_order(self, cs, c_order, comparison_cs, comparison_c_order)