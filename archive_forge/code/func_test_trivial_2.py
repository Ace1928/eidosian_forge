import random
import time
import networkx as nx
from networkx.algorithms.isomorphism.tree_isomorphism import (
from networkx.classes.function import is_directed
def test_trivial_2():
    print('trivial test 2')
    edges_1 = [('a', 'b'), ('a', 'c')]
    edges_2 = [('v', 'y')]
    t1 = nx.Graph()
    t1.add_edges_from(edges_1)
    t2 = nx.Graph()
    t2.add_edges_from(edges_2)
    isomorphism = tree_isomorphism(t1, t2)
    assert isomorphism == []