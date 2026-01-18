import math
import sys
from Bio import MissingPythonDependencyError
def to_networkx(tree):
    """Convert a Tree object to a networkx graph.

    The result is useful for graph-oriented analysis, and also interactive
    plotting with pylab, matplotlib or pygraphviz, though the resulting diagram
    is usually not ideal for displaying a phylogeny.

    Requires NetworkX version 0.99 or later.
    """
    try:
        import networkx
    except ImportError:
        raise MissingPythonDependencyError('Install NetworkX if you want to use to_networkx.') from None
    if networkx.__version__ >= '1.0':

        def add_edge(graph, n1, n2):
            graph.add_edge(n1, n2, weight=n2.branch_length or 1.0)
            if hasattr(n2, 'color') and n2.color is not None:
                graph[n1][n2]['color'] = n2.color.to_hex()
            elif hasattr(n1, 'color') and n1.color is not None:
                graph[n1][n2]['color'] = n1.color.to_hex()
                n2.color = n1.color
            if hasattr(n2, 'width') and n2.width is not None:
                graph[n1][n2]['width'] = n2.width
            elif hasattr(n1, 'width') and n1.width is not None:
                graph[n1][n2]['width'] = n1.width
                n2.width = n1.width
    elif networkx.__version__ >= '0.99':

        def add_edge(graph, n1, n2):
            graph.add_edge(n1, n2, n2.branch_length or 1.0)
    else:

        def add_edge(graph, n1, n2):
            graph.add_edge(n1, n2)

    def build_subgraph(graph, top):
        """Walk down the Tree, building graphs, edges and nodes."""
        for clade in top:
            graph.add_node(clade.root)
            add_edge(graph, top.root, clade.root)
            build_subgraph(graph, clade)
    if tree.rooted:
        G = networkx.DiGraph()
    else:
        G = networkx.Graph()
    G.add_node(tree.root)
    build_subgraph(G, tree.root)
    return G