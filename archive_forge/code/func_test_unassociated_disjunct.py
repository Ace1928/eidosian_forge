import logging
from os.path import abspath, dirname, join, normpath
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import build_model_size_report, log_model_size_report
from pyomo.common.fileutils import import_file
def test_unassociated_disjunct(self):
    m = ConcreteModel()
    m.x = Var(domain=Integers)
    m.d = Disjunct()
    m.d.c = Constraint(expr=m.x == 1)
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x == 5)
    m.disj = Disjunction(expr=[m.d2])
    model_size = build_model_size_report(m)
    self.assertEqual(model_size.warning.unassociated_disjuncts, 1)