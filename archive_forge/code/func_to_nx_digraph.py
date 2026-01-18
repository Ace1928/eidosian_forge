import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def to_nx_digraph(self, variables=None):
    """Convert this ``LazyArray`` into a ``networkx.DiGraph``."""
    import networkx as nx
    if variables is None:
        variables = set()
    elif isinstance(variables, LazyArray):
        variables = {variables}
    else:
        variables = set(variables)
    G = nx.DiGraph()
    nodemap = {}
    for i, node in enumerate(self.ascend()):
        nodemap[node] = i
        variable = node in variables or any((child in variables for child in node.deps))
        if variable:
            variables.add(node)
        G.add_node(i, array=node, variable=variable)
        for x in node.deps:
            G.add_edge(nodemap[x], nodemap[node])
    return G