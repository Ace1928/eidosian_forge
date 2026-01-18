import pytest
import networkx as nx
def test_raises_networkxerr():
    with pytest.raises(nx.NetworkXError):
        raise nx.NetworkXError