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
def test_unused_var(self):
    m = self.make_test_model()
    constraints = [m.eq1, m.eq2]
    variables = list(m.v.values())
    graph = get_bipartite_incidence_graph(variables, constraints)
    self.assertEqual(len(graph.nodes), 6)
    self.assertEqual(len(graph.edges), 4)
    self.assertTrue(nx.algorithms.bipartite.is_bipartite(graph))
    self.assertEqual(set(graph[0]), {2, 3})
    self.assertEqual(set(graph[1]), {2, 4})
    self.assertEqual(set(graph[2]), {0, 1})
    self.assertEqual(set(graph[3]), {0})
    self.assertEqual(set(graph[4]), {1})
    self.assertEqual(set(graph[5]), set())