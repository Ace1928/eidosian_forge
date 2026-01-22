import numpy as np
import heapq
def _invalidate_edge(graph, n1, n2):
    """Invalidates the edge (n1, n2) in the heap."""
    graph[n1][n2]['heap item'][3] = False