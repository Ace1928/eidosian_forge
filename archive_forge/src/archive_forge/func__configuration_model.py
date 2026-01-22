import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
def _configuration_model(deg_sequence, create_using, directed=False, in_deg_sequence=None, seed=None):
    """Helper function for generating either undirected or directed
    configuration model graphs.

    ``deg_sequence`` is a list of nonnegative integers representing the
    degree of the node whose label is the index of the list element.

    ``create_using`` see :func:`~networkx.empty_graph`.

    ``directed`` and ``in_deg_sequence`` are required if you want the
    returned graph to be generated using the directed configuration
    model algorithm. If ``directed`` is ``False``, then ``deg_sequence``
    is interpreted as the degree sequence of an undirected graph and
    ``in_deg_sequence`` is ignored. Otherwise, if ``directed`` is
    ``True``, then ``deg_sequence`` is interpreted as the out-degree
    sequence and ``in_deg_sequence`` as the in-degree sequence of a
    directed graph.

    .. note::

       ``deg_sequence`` and ``in_deg_sequence`` need not be the same
       length.

    ``seed`` is a random.Random or numpy.random.RandomState instance

    This function returns a graph, directed if and only if ``directed``
    is ``True``, generated according to the configuration model
    algorithm. For more information on the algorithm, see the
    :func:`configuration_model` or :func:`directed_configuration_model`
    functions.

    """
    n = len(deg_sequence)
    G = nx.empty_graph(n, create_using)
    if n == 0:
        return G
    if directed:
        pairs = zip_longest(deg_sequence, in_deg_sequence, fillvalue=0)
        out_deg, in_deg = zip(*pairs)
        out_stublist = _to_stublist(out_deg)
        in_stublist = _to_stublist(in_deg)
        seed.shuffle(out_stublist)
        seed.shuffle(in_stublist)
    else:
        stublist = _to_stublist(deg_sequence)
        n = len(stublist)
        half = n // 2
        seed.shuffle(stublist)
        out_stublist, in_stublist = (stublist[:half], stublist[half:])
    G.add_edges_from(zip(out_stublist, in_stublist))
    return G