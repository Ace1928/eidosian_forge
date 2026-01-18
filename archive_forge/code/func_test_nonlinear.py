import logging
from os.path import abspath, dirname, join, normpath
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import build_model_size_report, log_model_size_report
from pyomo.common.fileutils import import_file
def test_nonlinear(self):
    """Test nonlinear constraint detection."""
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.z.fix(3)
    m.c = Constraint(expr=m.x ** 2 == 4)
    m.c2 = Constraint(expr=m.x / m.y == 3)
    m.c3 = Constraint(expr=m.x * m.z == 5)
    m.c4 = Constraint(expr=m.x * m.y == 5)
    m.c4.deactivate()
    model_size = build_model_size_report(m)
    self.assertEqual(model_size.activated.nonlinear_constraints, 2)
    self.assertEqual(model_size.overall.nonlinear_constraints, 3)