from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def unfeasible():
    DG = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 2), (4, 5)])
    list(nx.all_topological_sorts(DG))