from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
class Test_Node:

    def __init__(self, n):
        self.label = n
        self.priority = 1

    def __repr__(self):
        return f'Node({self.label})'