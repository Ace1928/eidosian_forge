from itertools import combinations
import networkx as nx
from ..utils import arbitrary_element, not_implemented_for
def _simplegraph_eulerian_circuit(G, source):
    if G.is_directed():
        degree = G.out_degree
        edges = G.out_edges
    else:
        degree = G.degree
        edges = G.edges
    vertex_stack = [source]
    last_vertex = None
    while vertex_stack:
        current_vertex = vertex_stack[-1]
        if degree(current_vertex) == 0:
            if last_vertex is not None:
                yield (last_vertex, current_vertex)
            last_vertex = current_vertex
            vertex_stack.pop()
        else:
            _, next_vertex = arbitrary_element(edges(current_vertex))
            vertex_stack.append(next_vertex)
            G.remove_edge(current_vertex, next_vertex)