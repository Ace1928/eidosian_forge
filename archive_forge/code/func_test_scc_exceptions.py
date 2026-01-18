import random
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest
def test_scc_exceptions(self):
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from([(0, 1), (0, 2), (1, 2)])
    top_nodes = [0]
    msg = 'graph is not bipartite'
    with self.assertRaisesRegex(RuntimeError, msg):
        sccs = get_scc_of_projection(graph, top_nodes=top_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from([(0, 1), (0, 2)])
    top_nodes[0]
    msg = 'bipartite sets of different cardinalities'
    with self.assertRaisesRegex(RuntimeError, msg):
        sccs = get_scc_of_projection(graph, top_nodes=top_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1), (0, 2)])
    top_nodes = [0, 3]
    msg = 'without a perfect matching'
    with self.assertRaisesRegex(RuntimeError, msg):
        sccs = get_scc_of_projection(graph, top_nodes=top_nodes)