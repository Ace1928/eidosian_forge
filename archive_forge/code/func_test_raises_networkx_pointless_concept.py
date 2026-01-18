import pytest
import networkx as nx
def test_raises_networkx_pointless_concept():
    with pytest.raises(nx.NetworkXPointlessConcept):
        raise nx.NetworkXPointlessConcept