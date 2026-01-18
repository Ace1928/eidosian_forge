from io import StringIO
import pytest
import networkx as nx
from networkx.utils import graphs_equal
@pytest.mark.parametrize('G', (nx.Graph(), nx.DiGraph()))
@pytest.mark.parametrize('prog', ('neato', 'dot'))
def test_pydot(self, G, prog, tmp_path):
    """
        Validate :mod:`pydot`-based usage of the passed NetworkX graph with the
        passed basename of an external GraphViz command (e.g., `dot`, `neato`).
        """
    G.graph['name'] = 'G'
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('A', 'D')])
    G.add_node('E')
    graph_layout = nx.nx_pydot.pydot_layout(G, prog=prog)
    assert isinstance(graph_layout, dict)
    P = nx.nx_pydot.to_pydot(G)
    G2 = G.__class__(nx.nx_pydot.from_pydot(P))
    assert graphs_equal(G, G2)
    fname = tmp_path / 'out.dot'
    P.write_raw(fname)
    Pin_list = pydot.graph_from_dot_file(path=fname, encoding='utf-8')
    assert len(Pin_list) == 1
    Pin = Pin_list[0]
    n1 = sorted((p.get_name() for p in P.get_node_list()))
    n2 = sorted((p.get_name() for p in Pin.get_node_list()))
    assert n1 == n2
    e1 = sorted(((e.get_source(), e.get_destination()) for e in P.get_edge_list()))
    e2 = sorted(((e.get_source(), e.get_destination()) for e in Pin.get_edge_list()))
    assert e1 == e2
    Hin = nx.nx_pydot.read_dot(fname)
    Hin = G.__class__(Hin)
    assert graphs_equal(G, Hin)