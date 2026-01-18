from itertools import chain, combinations
import pytest
import networkx as nx
def test_termination():
    test1 = nx.karate_club_graph()
    test2 = nx.caveman_graph(2, 10)
    test2.add_edges_from([(0, 20), (20, 10)])
    nx.community.asyn_lpa_communities(test1)
    nx.community.asyn_lpa_communities(test2)