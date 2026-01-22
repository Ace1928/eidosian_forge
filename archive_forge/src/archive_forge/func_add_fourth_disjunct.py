from io import StringIO
import logging
from os.path import join, normpath
import pickle
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn
def add_fourth_disjunct(self, m):
    m.disjunction.deactivate()
    m.d4 = Disjunct()
    m.d4.x1_ub = Constraint(expr=m.x1 <= 8)
    m.d4.x2_lb = Constraint(expr=m.x2 >= -5)
    m.disjunction2 = Disjunction(expr=[m.d1, m.d2, m.d3, m.d4])