import math
from operator import itemgetter
import pytest
import networkx as nx
from networkx.algorithms.tree import branchings, recognition
def test_optimal_branching2b():
    G = build_branching(optimal_branching_2b)
    assert recognition.is_arborescence(G), True
    assert branchings.branching_weight(G) == 53