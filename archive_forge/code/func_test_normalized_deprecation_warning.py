import warnings
import pytest
import networkx as nx
def test_normalized_deprecation_warning():
    """Test that a deprecation warning is raised when s_metric is called with
    a `normalized` kwarg."""
    G = nx.cycle_graph(7)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert nx.s_metric(G) == 28
    with pytest.deprecated_call():
        nx.s_metric(G, normalized=True)
    with pytest.raises(TypeError):
        nx.s_metric(G, normalize=True)