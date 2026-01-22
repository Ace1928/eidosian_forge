import networkx as nx
from networkx import DiGraph, Graph, MultiDiGraph, MultiGraph, PlanarEmbedding
from networkx.classes.reportviews import NodeView
class LoopbackDispatcher:

    def __getattr__(self, item):
        try:
            return nx.utils.backends._registered_algorithms[item].orig_func
        except KeyError:
            raise AttributeError(item) from None

    @staticmethod
    def convert_from_nx(graph, *, edge_attrs=None, node_attrs=None, preserve_edge_attrs=None, preserve_node_attrs=None, preserve_graph_attrs=None, name=None, graph_name=None):
        if name in {'lexicographical_topological_sort', 'topological_generations', 'topological_sort', 'dfs_labeled_edges'}:
            return graph
        if isinstance(graph, NodeView):
            new_graph = Graph()
            new_graph.add_nodes_from(graph.items())
            graph = new_graph
            G = LoopbackGraph()
        elif not isinstance(graph, Graph):
            raise TypeError(f'Bad type for graph argument {graph_name} in {name}: {type(graph)}')
        elif graph.__class__ in {Graph, LoopbackGraph}:
            G = LoopbackGraph()
        elif graph.__class__ in {DiGraph, LoopbackDiGraph}:
            G = LoopbackDiGraph()
        elif graph.__class__ in {MultiGraph, LoopbackMultiGraph}:
            G = LoopbackMultiGraph()
        elif graph.__class__ in {MultiDiGraph, LoopbackMultiDiGraph}:
            G = LoopbackMultiDiGraph()
        elif graph.__class__ in {PlanarEmbedding, LoopbackPlanarEmbedding}:
            G = LoopbackDiGraph()
        else:
            G = graph.__class__()
        if preserve_graph_attrs:
            G.graph.update(graph.graph)
        if preserve_node_attrs:
            G.add_nodes_from(graph.nodes(data=True))
        elif node_attrs:
            G.add_nodes_from(((node, {k: datadict.get(k, default) for k, default in node_attrs.items() if default is not None or k in datadict}) for node, datadict in graph.nodes(data=True)))
        else:
            G.add_nodes_from(graph)
        if graph.is_multigraph():
            if preserve_edge_attrs:
                G.add_edges_from(((u, v, key, datadict) for u, nbrs in graph._adj.items() for v, keydict in nbrs.items() for key, datadict in keydict.items()))
            elif edge_attrs:
                G.add_edges_from(((u, v, key, {k: datadict.get(k, default) for k, default in edge_attrs.items() if default is not None or k in datadict}) for u, nbrs in graph._adj.items() for v, keydict in nbrs.items() for key, datadict in keydict.items()))
            else:
                G.add_edges_from(((u, v, key, {}) for u, nbrs in graph._adj.items() for v, keydict in nbrs.items() for key, datadict in keydict.items()))
        elif preserve_edge_attrs:
            G.add_edges_from(graph.edges(data=True))
        elif edge_attrs:
            G.add_edges_from(((u, v, {k: datadict.get(k, default) for k, default in edge_attrs.items() if default is not None or k in datadict}) for u, v, datadict in graph.edges(data=True)))
        else:
            G.add_edges_from(graph.edges)
        return G

    @staticmethod
    def convert_to_nx(obj, *, name=None):
        return obj

    @staticmethod
    def on_start_tests(items):
        for item in items:
            assert hasattr(item, 'add_marker')

    def can_run(self, name, args, kwargs):
        return hasattr(self, name)