import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_undirected(self):
    self.agraph_checks(nx.Graph())