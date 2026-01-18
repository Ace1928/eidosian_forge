import pytest
import networkx as nx
import networkx.algorithms.threshold as nxt
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_tg_creation_routines(self):
    s = nxt.left_d_threshold_sequence(5, 7)
    s = nxt.right_d_threshold_sequence(5, 7)
    s1 = nxt.swap_d(s, 1.0, 1.0)
    s1 = nxt.swap_d(s, 1.0, 1.0, seed=1)