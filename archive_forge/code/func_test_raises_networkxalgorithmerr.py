import pytest
import networkx as nx
def test_raises_networkxalgorithmerr():
    with pytest.raises(nx.NetworkXAlgorithmError):
        raise nx.NetworkXAlgorithmError