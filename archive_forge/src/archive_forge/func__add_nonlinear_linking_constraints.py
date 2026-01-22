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
def _add_nonlinear_linking_constraints(m):
    m.a = pyo.Var()
    m.b = pyo.Var()
    m.r = pyo.Var()
    n_inputs = 3

    def linking_constraint_rule(m, i):
        if i == 0:
            return m.a ** 2 - 0.5 * m.ex_block.inputs['input_0'] ** 2 == 0
        elif i == 1:
            return m.b ** 2 - 0.5 * m.ex_block.inputs['input_1'] ** 2 == 0
        elif i == 2:
            return m.r ** 2 - 0.5 * m.ex_block.inputs['input_2'] ** 2 == 0
    m.linking_constraint = pyo.Constraint(range(n_inputs), rule=linking_constraint_rule)