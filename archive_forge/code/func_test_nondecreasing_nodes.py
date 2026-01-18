from itertools import groupby
import pytest
import networkx as nx
from networkx import graph_atlas, graph_atlas_g
from networkx.generators.atlas import NUM_GRAPHS
from networkx.utils import edges_equal, nodes_equal, pairwise
def test_nondecreasing_nodes(self):
    for n1, n2 in pairwise(map(len, self.GAG)):
        assert n2 <= n1 + 1