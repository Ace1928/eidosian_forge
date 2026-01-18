import pytest
import networkx as nx
def test_raises_networkx_no_path():
    with pytest.raises(nx.NetworkXNoPath):
        raise nx.NetworkXNoPath