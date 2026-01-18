import pytest
import networkx as nx
def test_string_input():
    pytest.raises(nx.NetworkXException, nx.is_graphical, [], 'foo')
    pytest.raises(nx.NetworkXException, nx.is_graphical, ['red'], 'hh')
    pytest.raises(nx.NetworkXException, nx.is_graphical, ['red'], 'eg')