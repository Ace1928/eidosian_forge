from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
def _raise_on_directed(func):
    """
    A decorator which inspects the `create_using` argument and raises a
    NetworkX exception when `create_using` is a DiGraph (class or instance) for
    graph generators that do not support directed outputs.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get('create_using') is not None:
            G = nx.empty_graph(create_using=kwargs['create_using'])
            if G.is_directed():
                raise NetworkXError('Directed Graph not supported')
        return func(*args, **kwargs)
    return wrapper