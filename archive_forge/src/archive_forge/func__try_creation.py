import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
def _try_creation():
    edges = set()
    stubs = list(range(n)) * d
    while stubs:
        potential_edges = defaultdict(lambda: 0)
        seed.shuffle(stubs)
        stubiter = iter(stubs)
        for s1, s2 in zip(stubiter, stubiter):
            if s1 > s2:
                s1, s2 = (s2, s1)
            if s1 != s2 and (s1, s2) not in edges:
                edges.add((s1, s2))
            else:
                potential_edges[s1] += 1
                potential_edges[s2] += 1
        if not _suitable(edges, potential_edges):
            return None
        stubs = [node for node, potential in potential_edges.items() for _ in range(potential)]
    return edges