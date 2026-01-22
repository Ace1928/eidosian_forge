import itertools
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
import networkx as nx
from networkx.algorithms.approximation import local_node_connectivity
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
class _AntiGraph(nx.Graph):
    """
    Class for complement graphs.

    The main goal is to be able to work with big and dense graphs with
    a low memory footprint.

    In this class you add the edges that *do not exist* in the dense graph,
    the report methods of the class return the neighbors, the edges and
    the degree as if it was the dense graph. Thus it's possible to use
    an instance of this class with some of NetworkX functions. In this
    case we only use k-core, connected_components, and biconnected_components.
    """
    all_edge_dict = {'weight': 1}

    def single_edge_dict(self):
        return self.all_edge_dict
    edge_attr_dict_factory = single_edge_dict

    def __getitem__(self, n):
        """Returns a dict of neighbors of node n in the dense graph.

        Parameters
        ----------
        n : node
           A node in the graph.

        Returns
        -------
        adj_dict : dictionary
           The adjacency dictionary for nodes connected to n.

        """
        all_edge_dict = self.all_edge_dict
        return {node: all_edge_dict for node in set(self._adj) - set(self._adj[n]) - {n}}

    def neighbors(self, n):
        """Returns an iterator over all neighbors of node n in the
        dense graph.
        """
        try:
            return iter(set(self._adj) - set(self._adj[n]) - {n})
        except KeyError as err:
            raise NetworkXError(f'The node {n} is not in the graph.') from err

    class AntiAtlasView(Mapping):
        """An adjacency inner dict for AntiGraph"""

        def __init__(self, graph, node):
            self._graph = graph
            self._atlas = graph._adj[node]
            self._node = node

        def __len__(self):
            return len(self._graph) - len(self._atlas) - 1

        def __iter__(self):
            return (n for n in self._graph if n not in self._atlas and n != self._node)

        def __getitem__(self, nbr):
            nbrs = set(self._graph._adj) - set(self._atlas) - {self._node}
            if nbr in nbrs:
                return self._graph.all_edge_dict
            raise KeyError(nbr)

    class AntiAdjacencyView(AntiAtlasView):
        """An adjacency outer dict for AntiGraph"""

        def __init__(self, graph):
            self._graph = graph
            self._atlas = graph._adj

        def __len__(self):
            return len(self._atlas)

        def __iter__(self):
            return iter(self._graph)

        def __getitem__(self, node):
            if node not in self._graph:
                raise KeyError(node)
            return self._graph.AntiAtlasView(self._graph, node)

    @cached_property
    def adj(self):
        return self.AntiAdjacencyView(self)

    def subgraph(self, nodes):
        """This subgraph method returns a full AntiGraph. Not a View"""
        nodes = set(nodes)
        G = _AntiGraph()
        G.add_nodes_from(nodes)
        for n in G:
            Gnbrs = G.adjlist_inner_dict_factory()
            G._adj[n] = Gnbrs
            for nbr, d in self._adj[n].items():
                if nbr in G._adj:
                    Gnbrs[nbr] = d
                    G._adj[nbr][n] = d
        G.graph = self.graph
        return G

    class AntiDegreeView(nx.reportviews.DegreeView):

        def __iter__(self):
            all_nodes = set(self._succ)
            for n in self._nodes:
                nbrs = all_nodes - set(self._succ[n]) - {n}
                yield (n, len(nbrs))

        def __getitem__(self, n):
            nbrs = set(self._succ) - set(self._succ[n]) - {n}
            return len(nbrs) + (n in nbrs)

    @cached_property
    def degree(self):
        """Returns an iterator for (node, degree) and degree for single node.

        The node degree is the number of edges adjacent to the node.

        Parameters
        ----------
        nbunch : iterable container, optional (default=all nodes)
            A container of nodes.  The container will be iterated
            through once.

        weight : string or None, optional (default=None)
           The edge attribute that holds the numerical value used
           as a weight.  If None, then each edge has weight 1.
           The degree is the sum of the edge weights adjacent to the node.

        Returns
        -------
        deg:
            Degree of the node, if a single node is passed as argument.
        nd_iter : an iterator
            The iterator returns two-tuples of (node, degree).

        See Also
        --------
        degree

        Examples
        --------
        >>> G = nx.path_graph(4)
        >>> G.degree(0)  # node 0 with degree 1
        1
        >>> list(G.degree([0, 1]))
        [(0, 1), (1, 2)]

        """
        return self.AntiDegreeView(self)

    def adjacency(self):
        """Returns an iterator of (node, adjacency set) tuples for all nodes
           in the dense graph.

        This is the fastest way to look at every edge.
        For directed graphs, only outgoing adjacencies are included.

        Returns
        -------
        adj_iter : iterator
           An iterator of (node, adjacency set) for all nodes in
           the graph.

        """
        for n in self._adj:
            yield (n, set(self._adj) - set(self._adj[n]) - {n})