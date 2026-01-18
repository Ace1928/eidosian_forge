import pytest
import networkx as nx
from networkx.utils import pairwise
def test_others(self):
    assert nx.bellman_ford_path(self.XG, 's', 'v') == ['s', 'x', 'u', 'v']
    assert nx.bellman_ford_path_length(self.XG, 's', 'v') == 9
    assert nx.single_source_bellman_ford_path(self.XG, 's')['v'] == ['s', 'x', 'u', 'v']
    assert nx.single_source_bellman_ford_path_length(self.XG, 's')['v'] == 9
    D, P = nx.single_source_bellman_ford(self.XG, 's', target='v')
    assert D == 9
    assert P == ['s', 'x', 'u', 'v']
    P, D = nx.bellman_ford_predecessor_and_distance(self.XG, 's')
    assert P['v'] == ['u']
    assert D['v'] == 9
    P, D = nx.goldberg_radzik(self.XG, 's')
    assert P['v'] == 'u'
    assert D['v'] == 9