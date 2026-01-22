import networkx as nx
from networkx.algorithms.flow import (
from networkx.exception import NetworkXNoPath
from itertools import filterfalse as _filterfalse
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
def _unique_everseen(iterable):
    """List unique elements, preserving order. Remember all elements ever seen."""
    seen = set()
    seen_add = seen.add
    for element in _filterfalse(seen.__contains__, iterable):
        seen_add(element)
        yield element