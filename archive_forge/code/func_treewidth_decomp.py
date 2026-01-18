import itertools
import sys
from heapq import heapify, heappop, heappush
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch
def treewidth_decomp(G, heuristic=min_fill_in_heuristic):
    """Returns a treewidth decomposition using the passed heuristic.

    Parameters
    ----------
    G : NetworkX graph
    heuristic : heuristic function

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """
    graph = {n: set(G[n]) - {n} for n in G}
    node_stack = []
    elim_node = heuristic(graph)
    while elim_node is not None:
        nbrs = graph[elim_node]
        for u, v in itertools.permutations(nbrs, 2):
            if v not in graph[u]:
                graph[u].add(v)
        node_stack.append((elim_node, nbrs))
        for u in graph[elim_node]:
            graph[u].remove(elim_node)
        del graph[elim_node]
        elim_node = heuristic(graph)
    decomp = nx.Graph()
    first_bag = frozenset(graph.keys())
    decomp.add_node(first_bag)
    treewidth = len(first_bag) - 1
    while node_stack:
        curr_node, nbrs = node_stack.pop()
        old_bag = None
        for bag in decomp.nodes:
            if nbrs <= bag:
                old_bag = bag
                break
        if old_bag is None:
            old_bag = first_bag
        nbrs.add(curr_node)
        new_bag = frozenset(nbrs)
        treewidth = max(treewidth, len(new_bag) - 1)
        decomp.add_edge(old_bag, new_bag)
    return (treewidth, decomp)