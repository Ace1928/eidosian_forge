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
def test_subgraph_with_fewer_var_or_con(self):
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=[1, 2])
    m.v = pyo.Var(m.I)
    m.eq1 = pyo.Constraint(expr=m.v[1] + m.v[2] == 1)
    m.ineq1 = pyo.Constraint(expr=m.v[1] - m.v[2] <= 2)
    igraph = IncidenceGraphInterface(m, include_inequality=True)
    variables = list(m.v.values())
    constraints = [m.ineq1]
    matching = igraph.maximum_matching(variables, constraints)
    self.assertEqual(len(matching), 1)
    variables = [m.v[2]]
    constraints = [m.eq1, m.ineq1]
    matching = igraph.maximum_matching(variables, constraints)
    self.assertEqual(len(matching), 1)