from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import not_implemented_for, pairwise
def _all_simple_edge_paths(G, source, targets, cutoff):
    get_edges = (lambda node: G.edges(node, keys=True)) if G.is_multigraph() else lambda node: G.edges(node)
    current_path = {None: None}
    stack = [iter([(None, source)])]
    while stack:
        next_edge = next((e for e in stack[-1] if e[1] not in current_path), None)
        if next_edge is None:
            stack.pop()
            current_path.popitem()
            continue
        previous_node, next_node, *_ = next_edge
        if next_node in targets:
            yield (list(current_path.values()) + [next_edge])[2:]
        if len(current_path) - 1 < cutoff and targets - current_path.keys() - {next_node}:
            current_path[next_node] = next_edge
            stack.append(iter(get_edges(next_node)))