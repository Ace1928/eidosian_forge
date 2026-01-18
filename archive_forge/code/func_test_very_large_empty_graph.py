import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_very_large_empty_graph(self):
    G = nx.empty_graph(258049)
    result = BytesIO()
    nx.write_sparse6(G, result)
    assert result.getvalue() == b'>>sparse6<<:~~???~?@\n'