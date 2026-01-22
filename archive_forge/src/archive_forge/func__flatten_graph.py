import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatten.register(Graph)
def _flatten_graph(space, x) -> GraphInstance:
    """We're not using `.unflatten() for :class:`Box` and :class:`Discrete` because a graph is not a homogeneous space, see `.flatten` docstring."""

    def _graph_unflatten(unflatten_space, unflatten_x):
        ret = None
        if unflatten_space is not None and unflatten_x is not None:
            if isinstance(unflatten_space, Box):
                ret = unflatten_x.reshape(unflatten_x.shape[0], -1)
            elif isinstance(unflatten_space, Discrete):
                ret = np.zeros((unflatten_x.shape[0], unflatten_space.n - unflatten_space.start), dtype=unflatten_space.dtype)
                ret[np.arange(unflatten_x.shape[0]), unflatten_x - unflatten_space.start] = 1
        return ret
    nodes = _graph_unflatten(space.node_space, x.nodes)
    edges = _graph_unflatten(space.edge_space, x.edges)
    return GraphInstance(nodes, edges, x.edge_links)