from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
class TestTimeRespectingGraphMatcher:
    """
    A test class for the undirected temporal graph matcher.
    """

    def provide_g1_topology(self):
        G1 = nx.Graph()
        G1.add_edges_from(provide_g1_edgelist())
        return G1

    def provide_g2_path_3edges(self):
        G2 = nx.Graph()
        G2.add_edges_from([(0, 1), (1, 2), (2, 3)])
        return G2

    def test_timdelta_zero_timeRespecting_returnsTrue(self):
        G1 = self.provide_g1_topology()
        temporal_name = 'date'
        G1 = put_same_time(G1, temporal_name)
        G2 = self.provide_g2_path_3edges()
        d = timedelta()
        gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
        assert gm.subgraph_is_isomorphic()

    def test_timdelta_zero_datetime_timeRespecting_returnsTrue(self):
        G1 = self.provide_g1_topology()
        temporal_name = 'date'
        G1 = put_same_datetime(G1, temporal_name)
        G2 = self.provide_g2_path_3edges()
        d = timedelta()
        gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
        assert gm.subgraph_is_isomorphic()

    def test_attNameStrange_timdelta_zero_timeRespecting_returnsTrue(self):
        G1 = self.provide_g1_topology()
        temporal_name = 'strange_name'
        G1 = put_same_time(G1, temporal_name)
        G2 = self.provide_g2_path_3edges()
        d = timedelta()
        gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
        assert gm.subgraph_is_isomorphic()

    def test_notTimeRespecting_returnsFalse(self):
        G1 = self.provide_g1_topology()
        temporal_name = 'date'
        G1 = put_sequence_time(G1, temporal_name)
        G2 = self.provide_g2_path_3edges()
        d = timedelta()
        gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
        assert not gm.subgraph_is_isomorphic()

    def test_timdelta_one_config0_returns_no_embeddings(self):
        G1 = self.provide_g1_topology()
        temporal_name = 'date'
        G1 = put_time_config_0(G1, temporal_name)
        G2 = self.provide_g2_path_3edges()
        d = timedelta(days=1)
        gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
        count_match = len(list(gm.subgraph_isomorphisms_iter()))
        assert count_match == 0

    def test_timdelta_one_config1_returns_four_embedding(self):
        G1 = self.provide_g1_topology()
        temporal_name = 'date'
        G1 = put_time_config_1(G1, temporal_name)
        G2 = self.provide_g2_path_3edges()
        d = timedelta(days=1)
        gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
        count_match = len(list(gm.subgraph_isomorphisms_iter()))
        assert count_match == 4

    def test_timdelta_one_config2_returns_ten_embeddings(self):
        G1 = self.provide_g1_topology()
        temporal_name = 'date'
        G1 = put_time_config_2(G1, temporal_name)
        G2 = self.provide_g2_path_3edges()
        d = timedelta(days=1)
        gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
        L = list(gm.subgraph_isomorphisms_iter())
        count_match = len(list(gm.subgraph_isomorphisms_iter()))
        assert count_match == 10