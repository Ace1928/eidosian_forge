from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def add_entry(e):
    """Add a flow dict entry."""
    d = flow_dict[e[0]]
    for k in e[1:-2]:
        try:
            d = d[k]
        except KeyError:
            t = {}
            d[k] = t
            d = t
    d[e[-2]] = e[-1]