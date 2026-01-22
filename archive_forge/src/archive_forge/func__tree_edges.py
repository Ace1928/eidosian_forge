import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
def _tree_edges(n, r):
    if n == 0:
        return
    nodes = iter(range(n))
    parents = [next(nodes)]
    while parents:
        source = parents.pop(0)
        for i in range(r):
            try:
                target = next(nodes)
                parents.append(target)
                yield (source, target)
            except StopIteration:
                break