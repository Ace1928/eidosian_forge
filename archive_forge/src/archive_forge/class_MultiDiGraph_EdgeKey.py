import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
class MultiDiGraph_EdgeKey(nx.MultiDiGraph):
    """
    MultiDiGraph which assigns unique keys to every edge.

    Adds a dictionary edge_index which maps edge keys to (u, v, data) tuples.

    This is not a complete implementation. For Edmonds algorithm, we only use
    add_node and add_edge, so that is all that is implemented here. During
    additions, any specified keys are ignored---this means that you also
    cannot update edge attributes through add_node and add_edge.

    Why do we need this? Edmonds algorithm requires that we track edges, even
    as we change the head and tail of an edge, and even changing the weight
    of edges. We must reliably track edges across graph mutations.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        cls = super()
        cls.__init__(incoming_graph_data=incoming_graph_data, **attr)
        self._cls = cls
        self.edge_index = {}
        import warnings
        msg = 'MultiDiGraph_EdgeKey has been deprecated and will be removed in NetworkX 3.4.'
        warnings.warn(msg, DeprecationWarning)

    def remove_node(self, n):
        keys = set()
        for keydict in self.pred[n].values():
            keys.update(keydict)
        for keydict in self.succ[n].values():
            keys.update(keydict)
        for key in keys:
            del self.edge_index[key]
        self._cls.remove_node(n)

    def remove_nodes_from(self, nbunch):
        for n in nbunch:
            self.remove_node(n)

    def add_edge(self, u_for_edge, v_for_edge, key_for_edge, **attr):
        """
        Key is now required.

        """
        u, v, key = (u_for_edge, v_for_edge, key_for_edge)
        if key in self.edge_index:
            uu, vv, _ = self.edge_index[key]
            if u != uu or v != vv:
                raise Exception(f'Key {key!r} is already in use.')
        self._cls.add_edge(u, v, key, **attr)
        self.edge_index[key] = (u, v, self.succ[u][v][key])

    def add_edges_from(self, ebunch_to_add, **attr):
        for u, v, k, d in ebunch_to_add:
            self.add_edge(u, v, k, **d)

    def remove_edge_with_key(self, key):
        try:
            u, v, _ = self.edge_index[key]
        except KeyError as err:
            raise KeyError(f'Invalid edge key {key!r}') from err
        else:
            del self.edge_index[key]
            self._cls.remove_edge(u, v, key)

    def remove_edges_from(self, ebunch):
        raise NotImplementedError