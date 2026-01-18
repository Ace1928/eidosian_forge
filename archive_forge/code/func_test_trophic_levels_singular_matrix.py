import pytest
import networkx as nx
def test_trophic_levels_singular_matrix():
    """Should raise an error with graphs with only non-basal nodes"""
    matrix = np.identity(4)
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    with pytest.raises(nx.NetworkXError) as e:
        nx.trophic_levels(G)
    msg = 'Trophic levels are only defined for graphs where every node ' + 'has a path from a basal node (basal nodes are nodes with no ' + 'incoming edges).'
    assert msg in str(e.value)