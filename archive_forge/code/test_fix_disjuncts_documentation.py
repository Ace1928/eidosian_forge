import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.opt import check_available_solvers
It is expected that the result of this transformation is a MI(N)LP,
        so check that LogicalConstraints are handled correctly