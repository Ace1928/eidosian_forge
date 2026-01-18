import numpy as np
from collections import defaultdict
from ase.geometry.dimensionality.disjoint_set import DisjointSet
def traverse_component_graphs(adjacency):
    vertices = adjacency.keys()
    all_visited = {}
    ranks = {}
    for v in vertices:
        visited, rank = bfs(adjacency, v)
        all_visited[v] = visited
        ranks[v] = rank
    return (all_visited, ranks)