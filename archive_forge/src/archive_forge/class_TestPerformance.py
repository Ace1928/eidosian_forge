import pytest
import networkx as nx
from networkx import barbell_graph
from networkx.algorithms.community import modularity, partition_quality
from networkx.algorithms.community.quality import inter_community_edges
class TestPerformance:
    """Unit tests for the :func:`performance` function."""

    def test_bad_partition(self):
        """Tests that a poor partition has a low performance measure."""
        G = barbell_graph(3, 0)
        partition = [{0, 1, 4}, {2, 3, 5}]
        assert 8 / 15 == pytest.approx(partition_quality(G, partition)[1], abs=1e-07)

    def test_good_partition(self):
        """Tests that a good partition has a high performance measure."""
        G = barbell_graph(3, 0)
        partition = [{0, 1, 2}, {3, 4, 5}]
        assert 14 / 15 == pytest.approx(partition_quality(G, partition)[1], abs=1e-07)