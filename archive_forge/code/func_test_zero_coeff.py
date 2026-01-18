import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.dependencies import (
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_zero_coeff(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3])
    m.eq1 = pyo.Constraint(expr=m.x[1] + 0 * m.x[2] == 2)
    m.eq2 = pyo.Constraint(expr=m.x[1] ** 2 == 1)
    m.eq3 = pyo.Constraint(expr=m.x[2] * m.x[3] - m.x[1] == 1)
    igraph = IncidenceGraphInterface(m)
    var_dmp, con_dmp = igraph.dulmage_mendelsohn()
    self.assertGreater(len(var_dmp.unmatched), 0)