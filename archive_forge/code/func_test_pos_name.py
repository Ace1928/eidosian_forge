import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_pos_name(self):
    trgg = nx.thresholded_random_geometric_graph
    G = trgg(50, 0.25, 0.1, seed=42, pos_name='p', weight_name='wt')
    assert all((len(d['p']) == 2 for n, d in G.nodes.items()))
    assert all((d['wt'] > 0 for n, d in G.nodes.items()))