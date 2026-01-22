import logging
from .nesting import NestedState
from .diagrams_base import BaseGraph
def _get_subgraph(graph, name):
    """ Searches for subgraphs in a graph.
    Args:
        g (AGraph): Container to be searched.
        name (str): Name of the cluster.
    Returns: AGraph if a cluster called 'name' exists else None
    """
    sub_graph = graph.get_subgraph(name)
    if sub_graph:
        return sub_graph
    for sub in graph.subgraphs_iter():
        sub_graph = _get_subgraph(sub, name)
        if sub_graph:
            return sub_graph
    return None