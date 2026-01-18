import os
import tempfile
from io import StringIO
import pytest
import networkx as nx
from networkx.utils import graphs_equal
def test_read_write(self):
    G = nx.MultiGraph()
    G.graph['name'] = 'G'
    G.add_edge('1', '2', key='0')
    fh = StringIO()
    nx.nx_pydot.write_dot(G, fh)
    fh.seek(0)
    H = nx.nx_pydot.read_dot(fh)
    assert graphs_equal(G, H)