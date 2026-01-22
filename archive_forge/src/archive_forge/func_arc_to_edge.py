from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def arc_to_edge(self, G):
    """Returns a mapping from arcs to edges for a graph"""

    def fcn(G):
        res = ComponentMap()
        for edge in G.edges:
            arc = G.edges[edge]['arc']
            res[arc] = edge
        return res
    return self.cacher('arc_to_edge', fcn, G)