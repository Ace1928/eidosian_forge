import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
@unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
def test_optimize_no_decomposition(self):
    m = pyo.ConcreteModel()
    m.ex_block = ExternalGreyBoxBlock(concrete=True)
    block = m.ex_block
    m_ex = _make_external_model()
    input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
    external_vars = [m_ex.x, m_ex.y]
    residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
    external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
    ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons, solver_class=ImplicitFunctionSolver)
    block.set_external_model(ex_model)
    a = m.ex_block.inputs['input_0']
    b = m.ex_block.inputs['input_1']
    r = m.ex_block.inputs['input_2']
    x = m.ex_block.inputs['input_3']
    y = m.ex_block.inputs['input_4']
    m.obj = pyo.Objective(expr=(x - 2.0) ** 2 + (y - 2.0) ** 2 + (a - 2.0) ** 2 + (b - 2.0) ** 2 + (r - 2.0) ** 2)
    solver = pyo.SolverFactory('cyipopt')
    solver.solve(m)
    m_ex.obj = pyo.Objective(expr=(m_ex.x - 2.0) ** 2 + (m_ex.y - 2.0) ** 2 + (m_ex.a - 2.0) ** 2 + (m_ex.b - 2.0) ** 2 + (m_ex.r - 2.0) ** 2)
    m_ex.a.set_value(0.0)
    m_ex.b.set_value(0.0)
    m_ex.r.set_value(0.0)
    m_ex.y.set_value(0.0)
    m_ex.x.set_value(0.0)
    ipopt = pyo.SolverFactory('ipopt')
    ipopt.solve(m_ex)
    self.assertAlmostEqual(m_ex.a.value, a.value, delta=1e-08)
    self.assertAlmostEqual(m_ex.b.value, b.value, delta=1e-08)
    self.assertAlmostEqual(m_ex.r.value, r.value, delta=1e-08)
    self.assertAlmostEqual(m_ex.x.value, x.value, delta=1e-08)
    self.assertAlmostEqual(m_ex.y.value, y.value, delta=1e-08)