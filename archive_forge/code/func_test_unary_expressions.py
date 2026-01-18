from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_unary_expressions(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.a = Var()
    m.b = Var()
    m.c = Var()
    m.d = Var()
    m.c1 = Constraint(expr=0 <= sin(m.x))
    m.c2 = Constraint(expr=0 <= cos(m.y))
    m.c3 = Constraint(expr=0 <= tan(m.z))
    m.c4 = Constraint(expr=0 <= asin(m.a))
    m.c5 = Constraint(expr=0 <= acos(m.b))
    m.c6 = Constraint(expr=0 <= atan(m.c))
    m.c7 = Constraint(expr=0 <= sqrt(m.d))
    m.o = Objective(expr=m.x)
    self.assertTrue(satisfiable(m) is not False)