import logging
from .nesting import NestedState
from .diagrams_base import BaseGraph
 Searches for subgraphs in a graph.
    Args:
        g (AGraph): Container to be searched.
        name (str): Name of the cluster.
    Returns: AGraph if a cluster called 'name' exists else None
    