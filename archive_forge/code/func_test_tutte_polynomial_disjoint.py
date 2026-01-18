import pytest
import networkx as nx
@pytest.mark.parametrize('G', _test_tutte_graphs.keys())
def test_tutte_polynomial_disjoint(G):
    """Tutte polynomial factors into the Tutte polynomials of its components.
    Verify this property with the disjoint union of two copies of the input graph.
    """
    t_g = nx.tutte_polynomial(G)
    H = nx.disjoint_union(G, G)
    t_h = nx.tutte_polynomial(H)
    assert sympy.simplify(t_g * t_g).equals(t_h)