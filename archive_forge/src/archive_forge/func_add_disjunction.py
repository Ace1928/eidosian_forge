import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
def add_disjunction(self, b):
    m = b.model()
    b.another_disjunction = Disjunction(expr=[[(m.x[1] - 1) ** 2 + m.x[2] ** 2 <= 1], [-(m.x[1] - 2) ** 2 - (m.x[2] - 3) ** 2 >= -1]])