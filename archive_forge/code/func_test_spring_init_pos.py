import pytest
import networkx as nx
def test_spring_init_pos(self):
    import math
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    init_pos = {0: (0.0, 0.0)}
    fixed_pos = [0]
    pos = nx.fruchterman_reingold_layout(G, pos=init_pos, fixed=fixed_pos)
    has_nan = any((math.isnan(c) for coords in pos.values() for c in coords))
    assert not has_nan, 'values should not be nan'