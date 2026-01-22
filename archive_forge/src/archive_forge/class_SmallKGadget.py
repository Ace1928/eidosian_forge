import networkx as nx
from networkx.utils import not_implemented_for
class SmallKGadget:

    def __init__(self, k, degree, node, g):
        self.original = node
        self.k = k
        self.degree = degree
        self.g = g
        self.outer_vertices = [(node, x) for x in range(degree)]
        self.inner_vertices = [(node, x + degree) for x in range(degree)]
        self.core_vertices = [(node, x + 2 * degree) for x in range(k)]

    def replace_node(self):
        adj_view = self.g[self.original]
        for outer, inner, (neighbor, edge_attrs) in zip(self.outer_vertices, self.inner_vertices, list(adj_view.items())):
            self.g.add_edge(outer, inner)
            self.g.add_edge(outer, neighbor, **edge_attrs)
        for core in self.core_vertices:
            for inner in self.inner_vertices:
                self.g.add_edge(core, inner)
        self.g.remove_node(self.original)

    def restore_node(self):
        self.g.add_node(self.original)
        for outer in self.outer_vertices:
            adj_view = self.g[outer]
            for neighbor, edge_attrs in adj_view.items():
                if neighbor not in self.core_vertices:
                    self.g.add_edge(self.original, neighbor, **edge_attrs)
                    break
        self.g.remove_nodes_from(self.outer_vertices)
        self.g.remove_nodes_from(self.inner_vertices)
        self.g.remove_nodes_from(self.core_vertices)