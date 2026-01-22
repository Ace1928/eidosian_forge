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
def _create_pressure_drop_model(self):
    """
        Create a Pyomo model with pure ExternalGreyBoxModel embedded.
        """
    m = pyo.ConcreteModel()
    m.Pin = pyo.Var()
    m.c = pyo.Var()
    m.F = pyo.Var()
    m.P2 = pyo.Var()
    m.Pout = pyo.Var()
    m.Pin_con = pyo.Constraint(expr=m.Pin == 5.0)
    m.c_con = pyo.Constraint(expr=m.c == 1.0)
    m.F_con = pyo.Constraint(expr=m.F == 10.0)
    m.P2_con = pyo.Constraint(expr=m.P2 <= 5.0)
    m.obj = pyo.Objective(expr=(m.Pout - 3.0) ** 2)
    cons = [m.c_con, m.F_con, m.Pin_con, m.P2_con]
    inputs = [m.Pin, m.c, m.F]
    outputs = [m.P2, m.Pout]
    ex_model = PressureDropTwoOutputsWithHessian()
    m.egb = ExternalGreyBoxBlock()
    m.egb.set_external_model(ex_model, inputs=inputs, outputs=outputs)
    return m