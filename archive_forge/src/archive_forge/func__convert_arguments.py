import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
def _convert_arguments(self, backend_name, args, kwargs):
    """Convert graph arguments to the specified backend.

        Returns
        -------
        args tuple and kwargs dict
        """
    bound = self.__signature__.bind(*args, **kwargs)
    bound.apply_defaults()
    if not self.graphs:
        bound_kwargs = bound.kwargs
        del bound_kwargs['backend']
        return (bound.args, bound_kwargs)
    preserve_edge_attrs = self.preserve_edge_attrs
    edge_attrs = self.edge_attrs
    if preserve_edge_attrs is False:
        pass
    elif preserve_edge_attrs is True:
        edge_attrs = None
    elif isinstance(preserve_edge_attrs, str):
        if bound.arguments[preserve_edge_attrs] is True or callable(bound.arguments[preserve_edge_attrs]):
            preserve_edge_attrs = True
            edge_attrs = None
        elif bound.arguments[preserve_edge_attrs] is False and (isinstance(edge_attrs, str) and edge_attrs == preserve_edge_attrs or (isinstance(edge_attrs, dict) and preserve_edge_attrs in edge_attrs)):
            preserve_edge_attrs = False
            edge_attrs = None
        else:
            preserve_edge_attrs = False
    if edge_attrs is None:
        pass
    elif isinstance(edge_attrs, str):
        if edge_attrs[0] == '[':
            edge_attrs = {edge_attr: 1 for edge_attr in bound.arguments[edge_attrs[1:-1]]}
        elif callable(bound.arguments[edge_attrs]):
            preserve_edge_attrs = True
            edge_attrs = None
        elif bound.arguments[edge_attrs] is not None:
            edge_attrs = {bound.arguments[edge_attrs]: 1}
        elif self.name == 'to_numpy_array' and hasattr(bound.arguments['dtype'], 'names'):
            edge_attrs = {edge_attr: 1 for edge_attr in bound.arguments['dtype'].names}
        else:
            edge_attrs = None
    else:
        edge_attrs = {edge_attr: bound.arguments.get(val, 1) if isinstance(val, str) else val for key, val in edge_attrs.items() if (edge_attr := bound.arguments[key]) is not None}
    preserve_node_attrs = self.preserve_node_attrs
    node_attrs = self.node_attrs
    if preserve_node_attrs is False:
        pass
    elif preserve_node_attrs is True:
        node_attrs = None
    elif isinstance(preserve_node_attrs, str):
        if bound.arguments[preserve_node_attrs] is True or callable(bound.arguments[preserve_node_attrs]):
            preserve_node_attrs = True
            node_attrs = None
        elif bound.arguments[preserve_node_attrs] is False and (isinstance(node_attrs, str) and node_attrs == preserve_node_attrs or (isinstance(node_attrs, dict) and preserve_node_attrs in node_attrs)):
            preserve_node_attrs = False
            node_attrs = None
        else:
            preserve_node_attrs = False
    if node_attrs is None:
        pass
    elif isinstance(node_attrs, str):
        if node_attrs[0] == '[':
            node_attrs = {node_attr: None for node_attr in bound.arguments[node_attrs[1:-1]]}
        elif callable(bound.arguments[node_attrs]):
            preserve_node_attrs = True
            node_attrs = None
        elif bound.arguments[node_attrs] is not None:
            node_attrs = {bound.arguments[node_attrs]: None}
        else:
            node_attrs = None
    else:
        node_attrs = {node_attr: bound.arguments.get(val) if isinstance(val, str) else val for key, val in node_attrs.items() if (node_attr := bound.arguments[key]) is not None}
    preserve_graph_attrs = self.preserve_graph_attrs
    backend = _load_backend(backend_name)
    for gname in self.graphs:
        if gname in self.list_graphs:
            bound.arguments[gname] = [backend.convert_from_nx(g, edge_attrs=edge_attrs, node_attrs=node_attrs, preserve_edge_attrs=preserve_edge_attrs, preserve_node_attrs=preserve_node_attrs, preserve_graph_attrs=preserve_graph_attrs, name=self.name, graph_name=gname) if getattr(g, '__networkx_backend__', getattr(g, '__networkx_plugin__', 'networkx')) == 'networkx' else g for g in bound.arguments[gname]]
        else:
            graph = bound.arguments[gname]
            if graph is None:
                if gname in self.optional_graphs:
                    continue
                raise TypeError(f'Missing required graph argument `{gname}` in {self.name} function')
            if isinstance(preserve_edge_attrs, dict):
                preserve_edges = False
                edges = preserve_edge_attrs.get(gname, edge_attrs)
            else:
                preserve_edges = preserve_edge_attrs
                edges = edge_attrs
            if isinstance(preserve_node_attrs, dict):
                preserve_nodes = False
                nodes = preserve_node_attrs.get(gname, node_attrs)
            else:
                preserve_nodes = preserve_node_attrs
                nodes = node_attrs
            if isinstance(preserve_graph_attrs, set):
                preserve_graph = gname in preserve_graph_attrs
            else:
                preserve_graph = preserve_graph_attrs
            if getattr(graph, '__networkx_backend__', getattr(graph, '__networkx_plugin__', 'networkx')) == 'networkx':
                bound.arguments[gname] = backend.convert_from_nx(graph, edge_attrs=edges, node_attrs=nodes, preserve_edge_attrs=preserve_edges, preserve_node_attrs=preserve_nodes, preserve_graph_attrs=preserve_graph, name=self.name, graph_name=gname)
    bound_kwargs = bound.kwargs
    del bound_kwargs['backend']
    return (bound.args, bound_kwargs)