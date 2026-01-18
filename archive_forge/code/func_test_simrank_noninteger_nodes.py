import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
@pytest.mark.parametrize('simrank_similarity', simrank_algs)
def test_simrank_noninteger_nodes(self, simrank_similarity):
    G = nx.cycle_graph(5)
    G = nx.relabel_nodes(G, dict(enumerate('abcde')))
    expected = {'a': 1, 'b': 0.3951219505902448, 'c': 0.5707317069281646, 'd': 0.5707317069281646, 'e': 0.3951219505902449}
    actual = simrank_similarity(G, source='a')
    assert expected == pytest.approx(actual, abs=0.01)
    G = nx.DiGraph()
    G.add_node(0, label='Univ')
    G.add_node(1, label='ProfA')
    G.add_node(2, label='ProfB')
    G.add_node(3, label='StudentA')
    G.add_node(4, label='StudentB')
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (4, 2), (3, 0)])
    node_labels = dict(enumerate(nx.get_node_attributes(G, 'label').values()))
    G = nx.relabel_nodes(G, node_labels)
    expected = {'Univ': 1, 'ProfA': 0.0, 'ProfB': 0.1323363991265798, 'StudentA': 0.0, 'StudentB': 0.03387811817640443}
    actual = simrank_similarity(G, importance_factor=0.8, source='Univ')
    assert expected == pytest.approx(actual, abs=0.01)