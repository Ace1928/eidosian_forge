import networkx as nx
from networkx.utils import not_implemented_for
class LargeKGadget:

    def __init__(self, k, degree, node, g):
        self.original = node
        self.g = g
        self.k = k
        self.degree = degree
        self.outer_vertices = [(node, x) for x in range(degree)]
        self.core_vertices = [(node, x + degree) for x in range(degree - k)]

    def replace_node(self):
        adj_view = self.g[self.original]
        neighbors = list(adj_view.keys())
        edge_attrs = list(adj_view.values())
        for outer, neighbor, edge_attrs in zip(self.outer_vertices, neighbors, edge_attrs):
            self.g.add_edge(outer, neighbor, **edge_attrs)
        for core in self.core_vertices:
            for outer in self.outer_vertices:
                self.g.add_edge(core, outer)
        self.g.remove_node(self.original)

    def restore_node(self):
        self.g.add_node(self.original)
        for outer in self.outer_vertices:
            adj_view = self.g[outer]
            for neighbor, edge_attrs in list(adj_view.items()):
                if neighbor not in self.core_vertices:
                    self.g.add_edge(self.original, neighbor, **edge_attrs)
                    break
        g.remove_nodes_from(self.outer_vertices)
        g.remove_nodes_from(self.core_vertices)