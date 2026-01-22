import inspect
import itertools
import os
import warnings
from functools import partial
from importlib.metadata import entry_points
import networkx as nx
from .decorators import argmap
from .configs import Config, config
def _convert_graph(self, backend_name, graph, *, edge_attrs, node_attrs, preserve_edge_attrs, preserve_node_attrs, preserve_graph_attrs, graph_name, use_cache):
    if use_cache and (nx_cache := getattr(graph, '__networkx_cache__', None)) is not None:
        cache = nx_cache.setdefault('backends', {}).setdefault(backend_name, {})
        key = edge_key, node_key, graph_key = (frozenset(edge_attrs.items()) if edge_attrs is not None else preserve_edge_attrs, frozenset(node_attrs.items()) if node_attrs is not None else preserve_node_attrs, preserve_graph_attrs)
        if cache:
            warning_message = f'Using cached graph for {backend_name!r} backend in call to {self.name}.\n\nFor the cache to be consistent (i.e., correct), the input graph must not have been manually mutated since the cached graph was created. Examples of manually mutating the graph data structures resulting in an inconsistent cache include:\n\n    >>> G[u][v][key] = val\n\nand\n\n    >>> for u, v, d in G.edges(data=True):\n    ...     d[key] = val\n\nUsing methods such as `G.add_edge(u, v, weight=val)` will correctly clear the cache to keep it consistent. You may also use `G.__networkx_cache__.clear()` to manually clear the cache, or set `G.__networkx_cache__` to None to disable caching for G. Enable or disable caching via `nx.config.cache_converted_graphs` config.'
            for compat_key in itertools.product((edge_key, True) if edge_key is not True else (True,), (node_key, True) if node_key is not True else (True,), (graph_key, True) if graph_key is not True else (True,)):
                if (rv := cache.get(compat_key)) is not None:
                    warnings.warn(warning_message)
                    return rv
            if edge_key is not True and node_key is not True:
                for (ekey, nkey, gkey), val in list(cache.items()):
                    if edge_key is False or ekey is True:
                        pass
                    elif edge_key is True or ekey is False or (not edge_key.issubset(ekey)):
                        continue
                    if node_key is False or nkey is True:
                        pass
                    elif node_key is True or nkey is False or (not node_key.issubset(nkey)):
                        continue
                    if graph_key and (not gkey):
                        continue
                    warnings.warn(warning_message)
                    return val
    backend = _load_backend(backend_name)
    rv = backend.convert_from_nx(graph, edge_attrs=edge_attrs, node_attrs=node_attrs, preserve_edge_attrs=preserve_edge_attrs, preserve_node_attrs=preserve_node_attrs, preserve_graph_attrs=preserve_graph_attrs, name=self.name, graph_name=graph_name)
    if use_cache and nx_cache is not None:
        cache[key] = rv
        for cur_key in list(cache):
            if cur_key == key:
                continue
            ekey, nkey, gkey = cur_key
            if ekey is False or edge_key is True:
                pass
            elif ekey is True or edge_key is False or (not ekey.issubset(edge_key)):
                continue
            if nkey is False or node_key is True:
                pass
            elif nkey is True or node_key is False or (not nkey.issubset(node_key)):
                continue
            if gkey and (not graph_key):
                continue
            cache.pop(cur_key, None)
    return rv