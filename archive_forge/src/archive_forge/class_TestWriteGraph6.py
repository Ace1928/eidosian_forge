import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
class TestWriteGraph6:
    """Unit tests for writing a graph to a file in graph6 format."""

    def test_null_graph(self):
        result = BytesIO()
        nx.write_graph6(nx.null_graph(), result)
        assert result.getvalue() == b'>>graph6<<?\n'

    def test_trivial_graph(self):
        result = BytesIO()
        nx.write_graph6(nx.trivial_graph(), result)
        assert result.getvalue() == b'>>graph6<<@\n'

    def test_complete_graph(self):
        result = BytesIO()
        nx.write_graph6(nx.complete_graph(4), result)
        assert result.getvalue() == b'>>graph6<<C~\n'

    def test_large_complete_graph(self):
        result = BytesIO()
        nx.write_graph6(nx.complete_graph(67), result, header=False)
        assert result.getvalue() == b'~?@B' + b'~' * 368 + b'w\n'

    def test_no_header(self):
        result = BytesIO()
        nx.write_graph6(nx.complete_graph(4), result, header=False)
        assert result.getvalue() == b'C~\n'

    def test_complete_bipartite_graph(self):
        result = BytesIO()
        G = nx.complete_bipartite_graph(6, 9)
        nx.write_graph6(G, result, header=False)
        assert result.getvalue() == b'N??F~z{~Fw^_~?~?^_?\n'

    @pytest.mark.parametrize('G', (nx.MultiGraph(), nx.DiGraph()))
    def test_no_directed_or_multi_graphs(self, G):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.write_graph6(G, BytesIO())

    def test_length(self):
        for i in list(range(13)) + [31, 47, 62, 63, 64, 72]:
            g = nx.random_graphs.gnm_random_graph(i, i * i // 4, seed=i)
            gstr = BytesIO()
            nx.write_graph6(g, gstr, header=False)
            gstr = gstr.getvalue().rstrip()
            assert len(gstr) == ((i - 1) * i // 2 + 5) // 6 + (1 if i < 63 else 4)

    def test_roundtrip(self):
        for i in list(range(13)) + [31, 47, 62, 63, 64, 72]:
            G = nx.random_graphs.gnm_random_graph(i, i * i // 4, seed=i)
            f = BytesIO()
            nx.write_graph6(G, f)
            f.seek(0)
            H = nx.read_graph6(f)
            assert nodes_equal(G.nodes(), H.nodes())
            assert edges_equal(G.edges(), H.edges())

    def test_write_path(self):
        with tempfile.NamedTemporaryFile() as f:
            g6.write_graph6_file(nx.null_graph(), f)
            f.seek(0)
            assert f.read() == b'>>graph6<<?\n'

    @pytest.mark.parametrize('edge', ((0, 1), (1, 2), (1, 42)))
    def test_relabeling(self, edge):
        G = nx.Graph([edge])
        f = BytesIO()
        nx.write_graph6(G, f)
        f.seek(0)
        assert f.read() == b'>>graph6<<A_\n'